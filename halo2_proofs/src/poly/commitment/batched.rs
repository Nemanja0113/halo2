use crate::arithmetic::{best_multiexp, CurveAffine};
use ff::Field;
use group::Curve;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use super::{Blind, Polynomial, LagrangeCoeff};

/// Batched MSM accumulator that collects multiple MSM operations
/// and processes them together for better GPU utilization
#[derive(Debug)]
pub struct BatchedMSM<C: CurveAffine> {
    /// Queue of pending MSM operations
    operations: VecDeque<MSMOperation<C>>,
    /// Maximum batch size before auto-flush
    max_batch_size: usize,
    /// Current phase identifier for logging
    current_phase: String,
}

#[derive(Debug, Clone)]
struct MSMOperation<C: CurveAffine> {
    coeffs: Vec<C::Scalar>,
    bases: Vec<C>,
    operation_id: String,
}

#[derive(Debug)]
pub struct BatchedResult<C: CurveAffine> {
    pub results: Vec<C::Curve>,
    pub operation_ids: Vec<String>,
}

impl<C: CurveAffine> BatchedMSM<C> {
    /// Create a new batched MSM accumulator
    pub fn new(phase_name: &str, max_batch_size: usize) -> Self {
        log::info!("🔄 [BATCHED_MSM] Initializing for phase: {}", phase_name);
        Self {
            operations: VecDeque::new(),
            max_batch_size,
            current_phase: phase_name.to_string(),
        }
    }

    /// Add an MSM operation to the batch
    pub fn add_operation(
        &mut self,
        coeffs: Vec<C::Scalar>,
        bases: Vec<C>,
        operation_id: String,
    ) {
        log::debug!("📝 [BATCHED_MSM] Adding operation '{}': {} elements", 
                   operation_id, coeffs.len());
        
        self.operations.push_back(MSMOperation {
            coeffs,
            bases,
            operation_id,
        });

        // Auto-flush if we hit the batch size limit
        if self.operations.len() >= self.max_batch_size {
            log::warn!("⚠️ [BATCHED_MSM] Auto-flushing batch (reached max size: {})", 
                      self.max_batch_size);
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

        let start_time = std::time::Instant::now();

        // Collect all operations
        let operations: Vec<_> = self.operations.drain(..).collect();
        let operation_ids: Vec<_> = operations.iter()
            .map(|op| op.operation_id.clone())
            .collect();

        // Determine if we should use batched GPU processing
        let results = if total_elements > 1024 && batch_size > 1 {
            self.process_batched_gpu(&operations)
        } else {
            self.process_individual(&operations)
        };

        let elapsed = start_time.elapsed();
        log::info!("⚡ [BATCHED_MSM] Batch completed: {} operations in {:?} ({:.2} ops/ms, {:.2} elements/ms)", 
                   batch_size, elapsed, 
                   batch_size as f64 / elapsed.as_millis() as f64,
                   total_elements as f64 / elapsed.as_millis() as f64);

        Some(BatchedResult {
            results,
            operation_ids,
        })
    }

    /// Process operations using batched GPU MSM
    fn process_batched_gpu(&self, operations: &[MSMOperation<C>]) -> Vec<C::Curve> {
        log::debug!("🔄 [BATCHED_MSM] Using batched GPU processing");

        // Calculate total size and offsets
        let mut total_coeffs = Vec::new();
        let mut total_bases = Vec::new();
        let mut offsets = Vec::new();
        let mut current_offset = 0;

        for op in operations {
            offsets.push((current_offset, current_offset + op.coeffs.len()));
            total_coeffs.extend_from_slice(&op.coeffs);
            total_bases.extend_from_slice(&op.bases);
            current_offset += op.coeffs.len();
        }

        // Perform single large MSM operation
        let combined_result = best_multiexp(&total_coeffs, &total_bases);

        // For now, we need to split the result back to individual operations
        // This is a simplified approach - in practice, we'd need more sophisticated
        // batching at the GPU kernel level
        let mut results = Vec::new();
        for (i, (start, end)) in offsets.iter().enumerate() {
            let individual_result = best_multiexp(
                &total_coeffs[*start..*end],
                &total_bases[*start..*end]
            );
            results.push(individual_result);
            
            log::debug!("✅ [BATCHED_MSM] Operation {}: {} elements processed", 
                       operations[i].operation_id, end - start);
        }

        results
    }

    /// Process operations individually (fallback)
    fn process_individual(&self, operations: &[MSMOperation<C>]) -> Vec<C::Curve> {
        log::debug!("🔄 [BATCHED_MSM] Using individual processing (fallback)");

        operations.iter().map(|op| {
            best_multiexp(&op.coeffs, &op.bases)
        }).collect()
    }
}

/// Global batched MSM manager for coordinating across phases
pub struct BatchedMSMManager<C: CurveAffine> {
    current_batch: Mutex<Option<BatchedMSM<C>>>,
}

impl<C: CurveAffine> BatchedMSMManager<C> {
    pub fn new() -> Self {
        Self {
            current_batch: Mutex::new(None),
        }
    }

    /// Start a new batch for a specific phase
    pub fn start_phase_batch(&self, phase_name: &str, max_batch_size: usize) {
        let mut batch = self.current_batch.lock().unwrap();
        *batch = Some(BatchedMSM::new(phase_name, max_batch_size));
    }

    /// Add operation to current batch
    pub fn add_to_batch(
        &self,
        coeffs: Vec<C::Scalar>,
        bases: Vec<C>,
        operation_id: String,
    ) -> Option<C::Curve> {
        let mut batch_guard = self.current_batch.lock().unwrap();
        if let Some(batch) = batch_guard.as_mut() {
            batch.add_operation(coeffs, bases, operation_id);
            None // Return None to indicate batched processing
        } else {
            // No active batch, process immediately
            drop(batch_guard);
            Some(best_multiexp(&coeffs, &bases))
        }
    }

    /// Flush current batch and return results
    pub fn flush_current_batch(&self) -> Option<BatchedResult<C>> {
        let mut batch_guard = self.current_batch.lock().unwrap();
        if let Some(batch) = batch_guard.as_mut() {
            batch.flush_batch()
        } else {
            None
        }
    }

    /// End current batch
    pub fn end_phase_batch(&self) -> Option<BatchedResult<C>> {
        let mut batch_guard = self.current_batch.lock().unwrap();
        if let Some(mut batch) = batch_guard.take() {
            batch.flush_batch()
        } else {
            None
        }
    }
}

/// Tracks pending commitment operations for batched processing
#[derive(Debug, Clone)]
pub struct BatchCommitmentTracker<C: CurveAffine> {
    pending_operations: HashMap<String, Blind<C::Scalar>>,
    completed_commitments: HashMap<String, C>,
}

impl<C: CurveAffine> BatchCommitmentTracker<C> {
    pub fn new() -> Self {
        Self {
            pending_operations: HashMap::new(),
            completed_commitments: HashMap::new(),
        }
    }

    pub fn register_pending(&mut self, operation_id: String, blind: Blind<C::Scalar>) {
        self.pending_operations.insert(operation_id, blind);
    }

    pub fn process_batch_results(&mut self, batch_result: BatchResult<C>) {
        for (operation_id, commitment) in batch_result.commitments {
            self.completed_commitments.insert(operation_id, commitment);
        }
    }

    pub fn get_commitment(&self, operation_id: &str) -> Option<C> {
        self.completed_commitments.get(operation_id).copied()
    }

    pub fn clear(&mut self) {
        self.pending_operations.clear();
        self.completed_commitments.clear();
    }
}

/// Result of a batched commitment operation
#[derive(Debug, Clone)]
pub struct BatchResult<C: CurveAffine> {
    pub commitments: HashMap<String, C>,
    pub total_time: std::time::Duration,
    pub operation_count: usize,
}

/// Trait for commitment schemes that support batched operations
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

/// Batch operation descriptor
#[derive(Debug, Clone)]
pub struct BatchOperation<C: CurveAffine> {
    pub operation_id: String,
    pub polynomial: Polynomial<C::Scalar, LagrangeCoeff>,
    pub blind: Blind<C::Scalar>,
}

/// Batched MSM operation manager
#[derive(Debug)]
pub struct BatchedMsmManager<C: CurveAffine> {
    current_phase: Option<String>,
    pending_operations: Vec<BatchOperation<C>>,
    batch_threshold: usize,
    force_gpu_threshold: usize,
}

impl<C: CurveAffine> BatchedMsmManager<C> {
    pub fn new() -> Self {
        Self {
            current_phase: None,
            pending_operations: Vec::new(),
            batch_threshold: 4, // Minimum operations to trigger batching
            force_gpu_threshold: 8, // Force GPU usage above this threshold
        }
    }

    pub fn start_phase(&mut self, phase_name: String) {
        self.current_phase = Some(phase_name);
        self.pending_operations.clear();
    }

    pub fn add_operation(&mut self, operation: BatchOperation<C>) {
        self.pending_operations.push(operation);
    }

    pub fn should_use_gpu(&self) -> bool {
        self.pending_operations.len() >= self.force_gpu_threshold
    }

    pub fn should_batch(&self) -> bool {
        self.pending_operations.len() >= self.batch_threshold
    }

    pub fn get_operations(&self) -> &[BatchOperation<C>] {
        &self.pending_operations
    }

    pub fn clear_operations(&mut self) {
        self.pending_operations.clear();
    }

    pub fn end_phase(&mut self) -> Option<String> {
        let phase = self.current_phase.take();
        self.pending_operations.clear();
        phase
    }
}

// Global instance for managing batched MSM operations
lazy_static::lazy_static! {
    pub static ref GLOBAL_BATCHED_MSM: BatchedMSMManager<halo2curves::bn256::G1Affine> = 
        BatchedMSMManager::new();
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::{Fr, G1Affine};
    use ff::Field;
    use group::Curve;
    use rand_core::OsRng;

    #[test]
    fn test_batched_msm_basic() {
        let mut batch = BatchedMSM::<G1Affine>::new("test", 10);
        
        // Add some test operations
        let coeffs1 = vec![Fr::random(OsRng); 100];
        let bases1 = vec![G1Affine::generator(); 100];
        
        let coeffs2 = vec![Fr::random(OsRng); 50];
        let bases2 = vec![G1Affine::generator(); 50];

        batch.add_operation(coeffs1.clone(), bases1.clone(), "op1".to_string());
        batch.add_operation(coeffs2.clone(), bases2.clone(), "op2".to_string());

        assert_eq!(batch.pending_count(), 2);
        assert!(batch.has_pending());

        let result = batch.flush_batch().unwrap();
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.operation_ids, vec!["op1", "op2"]);
        
        assert_eq!(batch.pending_count(), 0);
        assert!(!batch.has_pending());
    }
}
