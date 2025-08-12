use crate::arithmetic::{best_multiexp, CurveAffine};
use ff::Field;
use group::Curve;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use crate::poly::{Polynomial, LagrangeCoeff};
use super::Blind;

use lazy_static::lazy_static;

/// Configuration for batched MSM operations
#[derive(Debug, Clone)]
pub struct BatchedMSMConfig {
    /// Enable/disable batching globally
    pub enabled: bool,
    /// Maximum number of operations to batch before auto-flush
    pub max_batch_size: usize,
    /// Minimum total elements to trigger GPU batching
    pub gpu_threshold: usize,
    /// Minimum batch size to trigger GPU batching
    pub gpu_batch_threshold: usize,
    /// Whether to force GPU usage for large batches
    pub force_gpu_for_large_batches: bool,
}

impl Default for BatchedMSMConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_batch_size: 16,
            gpu_threshold: 1024,
            gpu_batch_threshold: 4,
            force_gpu_for_large_batches: true,
        }
    }
}

/// Individual MSM operation to be batched
#[derive(Debug, Clone)]
pub struct MSMOperation<C: CurveAffine> {
    pub coeffs: Vec<C::Scalar>,
    pub bases: Vec<C>,
    pub operation_id: String,
    pub phase: String,
    pub priority: u8, // Higher number = higher priority
}

/// Result of a batched MSM operation
#[derive(Debug)]
pub struct BatchedResult<C: CurveAffine> {
    pub results: Vec<C::Curve>,
    pub operation_ids: Vec<String>,
    pub total_elements: usize,
    pub processing_time: Duration,
    pub used_gpu: bool,
}

/// Batched MSM accumulator that collects multiple MSM operations
/// and processes them together for better GPU utilization
#[derive(Debug)]
pub struct BatchedMSM<C: CurveAffine> {
    /// Queue of pending MSM operations
    operations: VecDeque<MSMOperation<C>>,
    /// Configuration for batching behavior
    config: BatchedMSMConfig,
    /// Current phase identifier for logging
    current_phase: String,
    /// Statistics tracking
    stats: BatchedMSMStats,
}

#[derive(Debug, Default, Clone)]
pub struct BatchedMSMStats {
    pub total_operations: usize,
    pub total_elements: usize,
    pub total_batches: usize,
    pub total_gpu_batches: usize,
    pub total_cpu_batches: usize,
    pub total_processing_time: Duration,
}

impl<C: CurveAffine> BatchedMSM<C> {
    /// Create a new batched MSM accumulator
    pub fn new(phase_name: &str, config: BatchedMSMConfig) -> Self {
        log::info!("🔄 [BATCHED_MSM] Initializing for phase: {} with config: {:?}", 
                   phase_name, config);
        Self {
            operations: VecDeque::new(),
            config,
            current_phase: phase_name.to_string(),
            stats: BatchedMSMStats::default(),
        }
    }

    /// Add an MSM operation to the batch
    pub fn add_operation(
        &mut self,
        coeffs: Vec<C::Scalar>,
        bases: Vec<C>,
        operation_id: String,
        priority: u8,
    ) {
        if !self.config.enabled {
            // If batching is disabled, process immediately
            let start = Instant::now();
            let result = best_multiexp(&coeffs, &bases);
            let elapsed = start.elapsed();
            log::debug!("⚡ [MSM_IMMEDIATE] {}: {} elements in {:?}", 
                       operation_id, coeffs.len(), elapsed);
            return;
        }

        log::debug!("📝 [BATCHED_MSM] Adding operation '{}': {} elements (priority: {})", 
                   operation_id, coeffs.len(), priority);
        
        self.stats.total_elements += coeffs.len();
        self.operations.push_back(MSMOperation {
            coeffs,
            bases,
            operation_id,
            phase: self.current_phase.clone(),
            priority,
        });

        self.stats.total_operations += 1;

        // Auto-flush if we hit the batch size limit
        if self.operations.len() >= self.config.max_batch_size {
            log::debug!("⚠️ [BATCHED_MSM] Auto-flushing batch (reached max size: {})", 
                      self.config.max_batch_size);
            self.flush_batch();
        }
    }

    /// Get the number of pending operations
    pub fn pending_count(&self) -> usize {
        self.operations.len()
    }

    /// Check if there are pending operations
    pub fn has_pending(&self) -> bool {
        !self.operations.is_empty()
    }

    /// Get current batch statistics
    pub fn get_stats(&self) -> &BatchedMSMStats {
        &self.stats
    }

    /// Process all pending MSM operations in a single batched call
    pub fn flush_batch(&mut self) -> Option<BatchedResult<C>> {
        if self.operations.is_empty() {
            return None;
        }

        let batch_size = self.operations.len();
        let total_elements: usize = self.operations.iter()
            .map(|op| op.coeffs.len())
            .sum();

        log::info!("🚀 [BATCHED_MSM] Processing batch for {}: {} operations, {} total elements", 
                   self.current_phase, batch_size, total_elements);

        let start_time = Instant::now();

        // Sort operations by priority (higher priority first)
        let mut operations: Vec<_> = self.operations.drain(..).collect();
        operations.sort_by(|a, b| b.priority.cmp(&a.priority));

        let operation_ids: Vec<_> = operations.iter()
            .map(|op| op.operation_id.clone())
            .collect();

        // Determine processing strategy
        let (results, used_gpu) = if self.should_use_gpu_batching(&operations, total_elements) {
            self.process_batched_gpu(&operations)
        } else {
            self.process_individual(&operations)
        };

        let processing_time = start_time.elapsed();
        
        // Update statistics
        self.stats.total_batches += 1;
        if used_gpu {
            self.stats.total_gpu_batches += 1;
        } else {
            self.stats.total_cpu_batches += 1;
        }
        self.stats.total_processing_time += processing_time;

        log::info!("✅ [BATCHED_MSM] Batch completed in {:?} (GPU: {})", 
                   processing_time, used_gpu);

        Some(BatchedResult {
            results,
            operation_ids,
            total_elements,
            processing_time,
            used_gpu,
        })
    }

    /// Determine if we should use GPU batching
    fn should_use_gpu_batching(&self, operations: &[MSMOperation<C>], total_elements: usize) -> bool {
        if !self.config.force_gpu_for_large_batches {
            return false;
        }

        let batch_size = operations.len();
        total_elements >= self.config.gpu_threshold && 
        batch_size >= self.config.gpu_batch_threshold
    }

    /// Process operations using batched GPU approach
    fn process_batched_gpu(&self, operations: &[MSMOperation<C>]) -> (Vec<C::Curve>, bool) {
        log::info!("🚀 [BATCHED_MSM_GPU] Processing {} operations on GPU", operations.len());
        
        // For now, fall back to individual processing
        // TODO: Implement actual GPU batching when icicle supports it
        let (results, _) = self.process_individual(operations);
        (results, false) // GPU not actually used yet
    }

    /// Process operations individually (fallback)
    fn process_individual(&self, operations: &[MSMOperation<C>]) -> (Vec<C::Curve>, bool) {
        log::debug!("🔄 [BATCHED_MSM_CPU] Processing {} operations individually", operations.len());
        
        let results = operations.iter().map(|op| {
            best_multiexp(&op.coeffs, &op.bases)
        }).collect();
        (results, false) // false indicates GPU was not used
    }

    /// Force flush all pending operations
    pub fn force_flush(&mut self) -> Option<BatchedResult<C>> {
        if self.operations.is_empty() {
            return None;
        }
        
        log::warn!("⚠️ [BATCHED_MSM] Force flushing {} pending operations", 
                   self.operations.len());
        self.flush_batch()
    }
}

/// Global batched MSM manager for coordinating operations across phases
#[derive(Debug)]
pub struct BatchedMsmManager<C: CurveAffine> {
    current_batch: Mutex<Option<BatchedMSM<C>>>,
    config: BatchedMSMConfig,
}

impl<C: CurveAffine> BatchedMsmManager<C> {
    /// Create a new global batched MSM manager
    pub fn new(config: BatchedMSMConfig) -> Self {
        log::info!("🔄 [BATCHED_MSM_MANAGER] Initializing with config: {:?}", config);
        Self {
            current_batch: Mutex::new(None),
            config,
        }
    }

    /// Start a new batch phase
    pub fn start_phase_batch(&self, phase_name: &str) {
        let mut current = self.current_batch.lock().unwrap();
        
        // Flush any existing batch
        if let Some(ref mut batch) = *current {
            if let Some(result) = batch.force_flush() {
                log::info!("🔄 [BATCHED_MSM_MANAGER] Flushed previous phase: {:?}", result);
            }
        }
        
        *current = Some(BatchedMSM::new(phase_name, self.config.clone()));
        log::info!("🔄 [BATCHED_MSM_MANAGER] Started new phase: {}", phase_name);
    }

    /// Add an operation to the current batch
    pub fn add_to_batch(
        &self,
        coeffs: Vec<C::Scalar>,
        bases: Vec<C>,
        operation_id: String,
        priority: u8,
    ) -> Option<C::Curve> {
        if !self.config.enabled {
            // If batching is disabled, process immediately
            return Some(best_multiexp(&coeffs, &bases));
        }

        let mut current = self.current_batch.lock().unwrap();
        if let Some(ref mut batch) = *current {
            batch.add_operation(coeffs, bases, operation_id, priority);
            None // Result will be available after flush
        } else {
            // No active batch, process immediately
            Some(best_multiexp(&coeffs, &bases))
        }
    }

    /// Flush the current batch
    pub fn flush_current_batch(&self) -> Option<BatchedResult<C>> {
        let mut current = self.current_batch.lock().unwrap();
        if let Some(ref mut batch) = *current {
            batch.flush_batch()
        } else {
            None
        }
    }

    /// End the current phase and flush
    pub fn end_phase_batch(&self) -> Option<BatchedResult<C>> {
        let mut current = self.current_batch.lock().unwrap();
        if let Some(ref mut batch) = *current {
            let result = batch.force_flush();
            *current = None;
            result
        } else {
            None
        }
    }

    /// Get current batch statistics
    pub fn get_stats(&self) -> Option<BatchedMSMStats> {
        let current = self.current_batch.lock().unwrap();
        current.as_ref().map(|batch| batch.get_stats().clone())
    }
}

/// Commitment tracker for batched operations
#[derive(Debug)]
pub struct BatchCommitmentTracker<C: CurveAffine> {
    pending_operations: HashMap<String, Blind<C::Scalar>>,
    completed_commitments: HashMap<String, C>,
}

impl<C: CurveAffine> BatchCommitmentTracker<C> {
    /// Create a new commitment tracker
    pub fn new() -> Self {
        Self {
            pending_operations: HashMap::new(),
            completed_commitments: HashMap::new(),
        }
    }

    /// Register a pending commitment operation
    pub fn register_pending(&mut self, operation_id: String, blind: Blind<C::Scalar>) {
        self.pending_operations.insert(operation_id, blind);
    }

    /// Process batch results and store commitments
    pub fn process_batch_results(&mut self, batch_result: BatchedResult<C>) {
        for (operation_id, commitment) in batch_result.operation_ids.iter()
            .zip(batch_result.results.iter()) {
            self.completed_commitments.insert(operation_id.clone(), commitment.to_affine());
        }
    }

    /// Get a completed commitment
    pub fn get_commitment(&self, operation_id: &str) -> Option<C> {
        self.completed_commitments.get(operation_id).copied()
    }

    /// Clear all tracked operations
    pub fn clear(&mut self) {
        self.pending_operations.clear();
        self.completed_commitments.clear();
    }
}

/// Result of a batch operation
#[derive(Debug)]
pub struct BatchResult<C: CurveAffine> {
    pub commitments: HashMap<String, C>,
    pub total_time: Duration,
    pub operation_count: usize,
}

/// Trait for prover parameters that support batched operations
pub trait BatchedParamsProver<C: CurveAffine> {
    /// Start a new batch phase for collecting operations
    fn start_batch_phase(&self, phase_name: &str);
    
    /// Add a Lagrange polynomial commitment to the current batch
    fn commit_lagrange_batched(
        &self,
        poly: &Polynomial<C::Scalar, LagrangeCoeff>,
        blind: Blind<C::Scalar>,
        operation_id: String,
    );
    
    /// Process all batched operations and return results
    fn end_batch_phase(&self) -> Option<BatchResult<C>>;
    
    /// Check if batching is currently active
    fn is_batching_active(&self) -> bool;
    
    /// Get the current batch size
    fn current_batch_size(&self) -> usize;
}

/// Individual batch operation
#[derive(Debug)]
pub struct BatchOperation<C: CurveAffine> {
    pub operation_id: String,
    pub polynomial: Polynomial<C::Scalar, LagrangeCoeff>,
    pub blind: Blind<C::Scalar>,
}

/// Global batched MSM manager instance
lazy_static::lazy_static! {
    static ref GLOBAL_BATCHED_MSM: Mutex<Option<BatchedMsmManager<halo2curves::bn256::G1Affine>>> = 
        Mutex::new(None);
}

/// Initialize the global batched MSM manager
pub fn init_global_batched_msm<C: CurveAffine>(config: BatchedMSMConfig) {
    let mut global = GLOBAL_BATCHED_MSM.lock().unwrap();
    *global = Some(BatchedMsmManager::new(config));
    log::info!("🔄 [GLOBAL_BATCHED_MSM] Initialized global manager");
}

/// Get the global batched MSM manager
pub fn get_global_batched_msm<C: CurveAffine>() -> Option<BatchedMsmManager<C>> {
    let global = GLOBAL_BATCHED_MSM.lock().unwrap();
    global.as_ref().map(|manager| {
        // This is a simplified approach - in practice you'd want proper type conversion
        BatchedMsmManager::new(BatchedMSMConfig::default())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::G1Affine;
    use ff::Field;

    #[test]
    fn test_batched_msm_basic() {
        let config = BatchedMSMConfig {
            enabled: true,
            max_batch_size: 4,
            gpu_threshold: 100,
            gpu_batch_threshold: 2,
            force_gpu_for_large_batches: false,
        };

        let mut batched_msm = BatchedMSM::<G1Affine>::new("test_phase", config);
        
        // Add some test operations
        for i in 0..3 {
            let coeffs = vec![G1Affine::Scalar::from(i as u64)];
            let bases = vec![G1Affine::generator()];
            batched_msm.add_operation(
                coeffs,
                bases,
                format!("test_op_{}", i),
                1,
            );
        }

        // Flush the batch
        let result = batched_msm.flush_batch();
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert_eq!(result.results.len(), 3);
        assert_eq!(result.operation_ids.len(), 3);
    }

    #[test]
    fn test_batched_msm_disabled() {
        let config = BatchedMSMConfig {
            enabled: false,
            ..Default::default()
        };

        let mut batched_msm = BatchedMSM::<G1Affine>::new("test_phase", config);
        
        // Operations should be processed immediately when batching is disabled
        let coeffs = vec![G1Affine::Scalar::from(1u64)];
        let bases = vec![G1Affine::generator()];
        batched_msm.add_operation(
            coeffs,
            bases,
            "test_op".to_string(),
            1,
        );

        // No pending operations when batching is disabled
        assert_eq!(batched_msm.pending_count(), 0);
    }
}
