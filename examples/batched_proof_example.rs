use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof_batched, keygen_pk, keygen_vk, Advice, Circuit, Column, ConstraintSystem,
        Error, Fixed, Instance,
    },
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::ProverSHPLONK,
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};
use halo2curves::bn256::{Bn256, Fr};
use rand_core::OsRng;

// Example circuit for demonstration
#[derive(Clone, Copy)]
struct ExampleCircuit {
    value: Option<Fr>,
}

impl Circuit<Fr> for ExampleCircuit {
    type Config = ExampleConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self { value: None }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let advice = meta.advice_column();
        let instance = meta.instance_column();
        let constant = meta.fixed_column();

        meta.enable_equality(advice);
        meta.enable_equality(instance);
        meta.enable_equality(constant);

        ExampleConfig {
            advice,
            instance,
            constant,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "example region",
            |mut region| {
                let value = self.value.unwrap_or(Fr::zero());
                region.assign_advice(|| "advice", config.advice, 0, || Value::known(value))?;
                region.assign_fixed(|| "fixed", config.constant, 0, || Value::known(Fr::one()))?;
                Ok(())
            },
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ExampleConfig {
    advice: Column<Advice>,
    instance: Column<Instance>,
    constant: Column<Fixed>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup parameters
    let k = 4; // 2^4 = 16 rows
    let params: ParamsKZG<Bn256> = ParamsKZG::setup(k, OsRng);
    
    // Create circuits
    let circuit1 = ExampleCircuit { value: Some(Fr::from(42)) };
    let circuit2 = ExampleCircuit { value: Some(Fr::from(84)) };
    let circuits = [circuit1, circuit2];
    
    // Generate keys
    let vk = keygen_vk(&params, &circuit1)?;
    let pk = keygen_pk(&params, vk, &circuit1)?;
    
    // Prepare instances (public inputs)
    let instances = [&[][..], &[][..]]; // No public inputs for this example
    let instance_refs: Vec<&[&[Fr]]> = instances.iter().map(|i| i.as_slice()).collect();
    
    // Create transcript
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    
    println!("🚀 Starting batched proof generation...");
    
    // Use the batched proof creation function
    let start = std::time::Instant::now();
    create_proof_batched::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<Bn256>, _, _, _, _>(
        &params,
        &pk,
        &circuits,
        &instance_refs,
        OsRng,
        &mut transcript,
    )?;
    
    let proof_time = start.elapsed();
    println!("✅ Batched proof generated in {:?}", proof_time);
    
    let proof = transcript.finalize();
    println!("📊 Proof size: {} bytes", proof.len());
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_vs_regular_proof() {
        let k = 4;
        let params: ParamsKZG<Bn256> = ParamsKZG::setup(k, OsRng);
        
        let circuit = ExampleCircuit { value: Some(Fr::from(42)) };
        let vk = keygen_vk(&params, &circuit).unwrap();
        let pk = keygen_pk(&params, vk, &circuit).unwrap();
        
        let instances = [&[][..]];
        let instance_refs: Vec<&[&[Fr]]> = instances.iter().map(|i| i.as_slice()).collect();
        
        // Test regular proof
        let mut transcript1 = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let start1 = std::time::Instant::now();
        halo2_proofs::plonk::create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<Bn256>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &instance_refs,
            OsRng,
            &mut transcript1,
        ).unwrap();
        let regular_time = start1.elapsed();
        
        // Test batched proof
        let mut transcript2 = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let start2 = std::time::Instant::now();
        create_proof_batched::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<Bn256>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &instance_refs,
            OsRng,
            &mut transcript2,
        ).unwrap();
        let batched_time = start2.elapsed();
        
        println!("Regular proof time: {:?}", regular_time);
        println!("Batched proof time: {:?}", batched_time);
        
        // Proofs should be identical
        assert_eq!(transcript1.finalize(), transcript2.finalize());
    }
}
