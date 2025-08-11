use halo2_proofs::{
    circuit::{SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, ConstraintSystem, Error},
    poly::{
        commitment::batched::{BatchedMSMConfig, BatchedMsmManager},
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};
use halo2curves::bn256::Bn256;
use rand_core::OsRng;

/// Simple circuit that demonstrates batched MSM operations
#[derive(Clone, Debug)]
struct BatchedMSMCircuit {
    a: Value<u64>,
    b: Value<u64>,
}

impl Circuit<halo2curves::bn256::Fr> for BatchedMSMCircuit {
    type Config = ();
    type FloorPlanner = SimpleFloorPlanner;

    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            a: Value::unknown(),
            b: Value::unknown(),
        }
    }

    fn configure(_meta: &mut ConstraintSystem<halo2curves::bn256::Fr>) -> Self::Config {
        ()
    }

    fn synthesize(
        &self,
        _config: Self::Config,
        _layouter: impl halo2_proofs::circuit::Layouter<halo2curves::bn256::Fr>,
    ) -> Result<(), Error> {
        // This is a simple circuit that doesn't use any advice columns
        // The batched MSM operations happen during proof generation
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up environment variables for batched MSM
    std::env::set_var("HALO2_MSM_BATCHING", "1");
    std::env::set_var("HALO2_MSM_BATCH_SIZE", "8");
    std::env::set_var("HALO2_MSM_GPU_THRESHOLD", "512");
    std::env::set_var("HALO2_MSM_GPU_BATCH_THRESHOLD", "2");
    std::env::set_var("HALO2_MSM_FORCE_GPU", "1");

    println!("🚀 [BATCHED_MSM_EXAMPLE] Starting batched MSM demonstration");

    // Create circuit instances
    let circuits = vec![
        BatchedMSMCircuit {
            a: Value::known(1),
            b: Value::known(2),
        },
        BatchedMSMCircuit {
            a: Value::known(3),
            b: Value::known(4),
        },
    ];

    // Generate parameters
    let params: ParamsKZG<Bn256> = ParamsKZG::setup(8, OsRng);
    println!("✅ [BATCHED_MSM_EXAMPLE] Generated parameters");

    // Create proving key
    let vk = halo2_proofs::plonk::keygen_vk(&params, &circuits[0])?;
    let pk = halo2_proofs::plonk::keygen_pk(&params, vk, &circuits[0])?;
    println!("✅ [BATCHED_MSM_EXAMPLE] Generated proving key");

    // Create transcript
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    // Create proof with batched MSM enabled
    println!("🔄 [BATCHED_MSM_EXAMPLE] Creating proof with batched MSM...");
    let start = std::time::Instant::now();
    
    halo2_proofs::plonk::create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &circuits,
        &[&[], &[]], // No public inputs
        OsRng,
        &mut transcript,
    )?;

    let elapsed = start.elapsed();
    println!("✅ [BATCHED_MSM_EXAMPLE] Proof created in {:?}", elapsed);

    // Get the proof bytes
    let proof = transcript.finalize();
    println!("📊 [BATCHED_MSM_EXAMPLE] Proof size: {} bytes", proof.len());

    // Demonstrate manual batched MSM usage
    println!("\n🔄 [BATCHED_MSM_EXAMPLE] Demonstrating manual batched MSM usage...");
    
    let config = BatchedMSMConfig {
        enabled: true,
        max_batch_size: 4,
        gpu_threshold: 256,
        gpu_batch_threshold: 2,
        force_gpu_for_large_batches: true,
    };

    let mut batched_manager = BatchedMsmManager::new(config);
    
    // Start a phase
    batched_manager.start_phase_batch("manual_demo");
    
    // Add some operations
    for i in 0..6 {
        let coeffs = vec![halo2curves::bn256::Fr::from(i as u64)];
        let bases = vec![halo2curves::bn256::G1Affine::generator()];
        
        batched_manager.add_to_batch(
            coeffs,
            bases,
            format!("demo_op_{}", i),
            1,
        );
    }
    
    // Flush the batch
    if let Some(result) = batched_manager.flush_current_batch() {
        println!("✅ [BATCHED_MSM_EXAMPLE] Manual batch completed:");
        println!("   - Operations: {}", result.operation_ids.len());
        println!("   - Total elements: {}", result.total_elements);
        println!("   - Processing time: {:?}", result.processing_time);
        println!("   - Used GPU: {}", result.used_gpu);
    }
    
    // End the phase
    if let Some(result) = batched_manager.end_phase_batch() {
        println!("✅ [BATCHED_MSM_EXAMPLE] Phase completed with {} operations", 
                 result.operation_ids.len());
    }

    println!("\n🎉 [BATCHED_MSM_EXAMPLE] Demonstration completed successfully!");
    println!("💡 [BATCHED_MSM_EXAMPLE] Check the logs above for detailed batched MSM statistics");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_msm_example() {
        // This test ensures the example compiles and runs
        let result = main();
        assert!(result.is_ok(), "Batched MSM example should run successfully");
    }
} 