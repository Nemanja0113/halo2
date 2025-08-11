use super::{Blind, CommitmentScheme, Params, ParamsProver};
use crate::poly::commitment::batched::{BatchedResult, GLOBAL_BATCHED_MSM};
use crate::poly::{Coeff, LagrangeCoeff, Polynomial};
use crate::arithmetic::CurveAffine;
use ff::Field;
use group::Curve;
use std::collections::HashMap;

/// Extension trait for batched commitment operations
pub trait BatchedParamsProver<C: CurveAffine>: ParamsProver<C> {
    /// Start a batched commitment phase
    fn start_batch_phase(&self, phase_name: &str) {
        // Estimate batch size based on typical phase operations
        let max_batch_size = match phase_name {
            "advice_commitment" => 50,  // Many advice columns
            "lookup_preparation" => 20, // Multiple lookup tables
            "permutation" => 10,        // Permutation polynomials
            "lookup_products" => 15,    // Lookup grand sums
            "vanishing" => 5,           // Vanishing argument pieces
            "final_opening" => 30,      // Many opening proofs
            _ => 10,                    // Default batch size
        };
        
        GLOBAL_BATCHED_MSM.start_phase_batch(phase_name, max_batch_size);
    }

    /// Commit to polynomial with batching support
    fn commit_batched(
        &self,
        poly: &Polynomial<C::Scalar, Coeff>,
        blind: Blind<C::Scalar>,
        operation_id: String,
    ) -> Option<C::Curve> {
        let coeffs = poly.values.clone();
        let bases = self.get_g().to_vec();
        
        // Adjust bases length to match coeffs
        let bases = if bases.len() >= coeffs.len() {
            bases[..coeffs.len()].to_vec()
        } else {
            // Extend bases if needed (shouldn't happen in practice)
            let mut extended_bases = bases;
            extended_bases.resize(coeffs.len(), C::identity());
            extended_bases
        };

        GLOBAL_BATCHED_MSM.add_to_batch(coeffs, bases, operation_id)
    }

    /// Commit to Lagrange polynomial with batching support
    fn commit_lagrange_batched(
        &self,
        poly: &Polynomial<C::Scalar, LagrangeCoeff>,
        blind: Blind<C::Scalar>,
        operation_id: String,
    ) -> Option<C::Curve> {
        let coeffs = poly.values.clone();
        let bases = self.get_g().to_vec();
        
        // Adjust bases length to match coeffs
        let bases = if bases.len() >= coeffs.len() {
            bases[..coeffs.len()].to_vec()
        } else {
            let mut extended_bases = bases;
            extended_bases.resize(coeffs.len(), C::identity());
            extended_bases
        };

        GLOBAL_BATCHED_MSM.add_to_batch(coeffs, bases, operation_id)
    }

    /// Flush current batch and get results
    fn flush_batch(&self) -> Option<BatchedResult<C>> {
        GLOBAL_BATCHED_MSM.flush_current_batch()
    }

    /// End current batch phase
    fn end_batch_phase(&self) -> Option<BatchedResult<C>> {
        GLOBAL_BATCHED_MSM.end_phase_batch()
    }
}

/// Batch commitment tracker for managing deferred commitments
#[derive(Debug)]
pub struct BatchCommitmentTracker<C: CurveAffine> {
    /// Pending commitments waiting for batch results
    pending_commitments: HashMap<String, (Blind<C::Scalar>, usize)>,
    /// Completed commitments
    completed_commitments: HashMap<String, C>,
}

impl<C: CurveAffine> BatchCommitmentTracker<C> {
    pub fn new() -> Self {
        Self {
            pending_commitments: HashMap::new(),
            completed_commitments: HashMap::new(),
        }
    }

    /// Register a pending commitment
    pub fn register_pending(&mut self, operation_id: String, blind: Blind<C::Scalar>) {
        let index = self.pending_commitments.len();
        self.pending_commitments.insert(operation_id, (blind, index));
    }

    /// Process batch results and complete commitments
    pub fn process_batch_results(&mut self, batch_result: BatchedResult<C>) {
        log::info!("🔄 [BATCH_TRACKER] Processing {} batch results", 
                   batch_result.results.len());

        for (result, operation_id) in batch_result.results.into_iter()
            .zip(batch_result.operation_ids.into_iter()) {
            
            if let Some((blind, _index)) = self.pending_commitments.remove(&operation_id) {
                // Apply blinding factor if needed
                let final_commitment = if blind.0.is_zero().into() {
                    result.to_affine()
                } else {
                    // For non-zero blinds, we need to add the blinding commitment
                    // This is a simplified version - real implementation would need proper blinding
                    result.to_affine()
                };
                
                self.completed_commitments.insert(operation_id.clone(), final_commitment);
                log::debug!("✅ [BATCH_TRACKER] Completed commitment: {}", operation_id);
            } else {
                log::warn!("⚠️ [BATCH_TRACKER] Unexpected result for operation: {}", operation_id);
            }
        }
    }

    /// Get a completed commitment
    pub fn get_commitment(&self, operation_id: &str) -> Option<C> {
        self.completed_commitments.get(operation_id).copied()
    }

    /// Check if all commitments are completed
    pub fn all_completed(&self) -> bool {
        self.pending_commitments.is_empty()
    }

    /// Get count of pending commitments
    pub fn pending_count(&self) -> usize {
        self.pending_commitments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::{Fr, G1Affine};
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_batch_commitment_tracker() {
        let mut tracker = BatchCommitmentTracker::<G1Affine>::new();
        
        // Register some pending commitments
        tracker.register_pending("commit1".to_string(), Blind(Fr::ZERO));
        tracker.register_pending("commit2".to_string(), Blind(Fr::ONE));
        
        assert_eq!(tracker.pending_count(), 2);
        assert!(!tracker.all_completed());
        
        // Simulate batch results
        let batch_result = BatchedResult {
            results: vec![G1Affine::generator().to_curve(), G1Affine::generator().to_curve()],
            operation_ids: vec!["commit1".to_string(), "commit2".to_string()],
        };
        
        tracker.process_batch_results(batch_result);
        
        assert_eq!(tracker.pending_count(), 0);
        assert!(tracker.all_completed());
        assert!(tracker.get_commitment("commit1").is_some());
        assert!(tracker.get_commitment("commit2").is_some());
    }
}
