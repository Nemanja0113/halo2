use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{KZGCommitmentScheme, ParamsKZG};
use crate::arithmetic::CurveAffine;
use crate::poly::commitment::{
    batched::{BatchResult, BatchedParamsProver, BatchOperation, BatchedMsmManager},
    Blind, Polynomial, LagrangeCoeff,
};

/// Batched implementation for KZG commitment scheme
impl<E: pairing::Engine> BatchedParamsProver<E::G1Affine> for ParamsKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr>,
{
    fn start_batch_phase(&self, phase_name: &str) {
        // Initialize batching for this phase
        log::debug!("🔄 Starting batch phase: {}", phase_name);
        
        // In a real implementation, this would initialize GPU context
        // and prepare for batched operations
        if let Some(manager) = self.get_batch_manager() {
            let mut manager = manager.lock().unwrap();
            manager.start_phase(phase_name.to_string());
        }
    }

    fn commit_lagrange_batched(
        &self,
        poly: &Polynomial<E::Fr, LagrangeCoeff>,
        blind: Blind<E::Fr>,
        operation_id: String,
    ) {
        if let Some(manager) = self.get_batch_manager() {
            let mut manager = manager.lock().unwrap();
            manager.add_operation(BatchOperation {
                operation_id,
                polynomial: poly.clone(),
                blind,
            });
        }
    }

    fn end_batch_phase(&self) -> Option<BatchResult<E::G1Affine>> {
        let start_time = Instant::now();
        
        if let Some(manager) = self.get_batch_manager() {
            let mut manager = manager.lock().unwrap();
            let operations = manager.get_operations();
            
            if operations.is_empty() {
                manager.end_phase();
                return None;
            }

            log::info!("🚀 Processing {} batched operations", operations.len());

            let mut commitments = HashMap::new();
            
            if manager.should_use_gpu() && operations.len() > 1 {
                // Use batched GPU MSM
                log::debug!("Using batched GPU MSM for {} operations", operations.len());
                
                // Collect all polynomials and blinds for batched processing
                let polys: Vec<_> = operations.iter().map(|op| &op.polynomial).collect();
                let blinds: Vec<_> = operations.iter().map(|op| op.blind).collect();
                
                // Perform batched MSM operation
                let batch_commitments = self.commit_lagrange_batch(&polys, &blinds);
                
                // Map results back to operation IDs
                for (i, op) in operations.iter().enumerate() {
                    if let Some(commitment) = batch_commitments.get(i) {
                        commitments.insert(op.operation_id.clone(), *commitment);
                    }
                }
            } else {
                // Fall back to individual operations
                log::debug!("Using individual operations for {} commitments", operations.len());
                
                for op in operations {
                    let commitment = self.commit_lagrange(&op.polynomial, op.blind);
                    commitments.insert(op.operation_id.clone(), commitment);
                }
            }

            let total_time = start_time.elapsed();
            let operation_count = operations.len();
            
            manager.clear_operations();
            manager.end_phase();

            log::info!("✅ Batch completed: {} ops in {:?} ({:.2}ms/op)", 
                      operation_count, total_time, 
                      total_time.as_millis() as f64 / operation_count as f64);

            Some(BatchResult {
                commitments,
                total_time,
                operation_count,
            })
        } else {
            None
        }
    }

    fn is_batching_active(&self) -> bool {
        if let Some(manager) = self.get_batch_manager() {
            let manager = manager.lock().unwrap();
            manager.current_phase.is_some()
        } else {
            false
        }
    }

    fn current_batch_size(&self) -> usize {
        if let Some(manager) = self.get_batch_manager() {
            let manager = manager.lock().unwrap();
            manager.pending_operations.len()
        } else {
            0
        }
    }
}

impl<E: pairing::Engine> ParamsKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr>,
{
    /// Get or create the batch manager for this params instance
    fn get_batch_manager(&self) -> Option<Arc<Mutex<BatchedMsmManager<E::G1Affine>>>> {
        // In a real implementation, this would be stored as a field in ParamsKZG
        // For now, we'll use a thread-local or global manager
        thread_local! {
            static BATCH_MANAGER: Arc<Mutex<BatchedMsmManager<E::G1Affine>>> = 
                Arc::new(Mutex::new(BatchedMsmManager::new()));
        }
        
        BATCH_MANAGER.with(|manager| Some(manager.clone()))
    }

    /// Perform batched Lagrange commitment using GPU MSM
    fn commit_lagrange_batch(
        &self,
        polys: &[&Polynomial<E::Fr, LagrangeCoeff>],
        blinds: &[Blind<E::Fr>],
    ) -> Vec<E::G1Affine> {
        let start = Instant::now();
        
        // Prepare data for batched MSM
        let mut all_bases = Vec::new();
        let mut all_scalars = Vec::new();
        let mut batch_sizes = Vec::new();
        
        for (poly, blind) in polys.iter().zip(blinds.iter()) {
            let n = poly.len();
            batch_sizes.push(n + 1); // +1 for blind
            
            // Add polynomial coefficients
            all_scalars.extend(poly.iter());
            all_bases.extend(self.g.iter().take(n));
            
            // Add blind factor
            all_scalars.push(blind.0);
            all_bases.push(self.h);
        }

        log::debug!("Batched MSM: {} total scalars across {} polynomials", 
                   all_scalars.len(), polys.len());

        // Perform batched MSM operation
        let commitments = if cfg!(feature = "gpu") {
            self.msm_gpu_batched(&all_bases, &all_scalars, &batch_sizes)
        } else {
            self.msm_cpu_batched(&all_bases, &all_scalars, &batch_sizes)
        };

        log::debug!("Batched MSM completed in {:?}", start.elapsed());
        commitments
    }

    /// GPU-accelerated batched MSM
    #[cfg(feature = "gpu")]
    fn msm_gpu_batched(
        &self,
        bases: &[E::G1Affine],
        scalars: &[E::Fr],
        batch_sizes: &[usize],
    ) -> Vec<E::G1Affine> {
        use crate::icicle_batched::batched_msm_gpu;
        
        match batched_msm_gpu(bases, scalars, batch_sizes) {
            Ok(results) => results,
            Err(e) => {
                log::warn!("GPU batched MSM failed: {:?}, falling back to CPU", e);
                self.msm_cpu_batched(bases, scalars, batch_sizes)
            }
        }
    }

    /// CPU fallback for batched MSM
    fn msm_cpu_batched(
        &self,
        bases: &[E::G1Affine],
        scalars: &[E::Fr],
        batch_sizes: &[usize],
    ) -> Vec<E::G1Affine> {
        let mut results = Vec::new();
        let mut offset = 0;
        
        for &size in batch_sizes {
            let batch_bases = &bases[offset..offset + size];
            let batch_scalars = &scalars[offset..offset + size];
            
            // Perform individual MSM for this batch
            let result = crate::arithmetic::best_multiexp(batch_scalars, batch_bases);
            results.push(result.to_affine());
            
            offset += size;
        }
        
        results
    }
}
