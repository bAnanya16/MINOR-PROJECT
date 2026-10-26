# Lifelong MAPF Disruption Repair: Benchmark Evaluation

| Experimental Condition | Overlap Mode | Incidents | Success Rate | Mean Recovery (steps) | Cost Δ (%) | LLM Calls | Cache Hit Rate (%) | Verified Reuse (%) | Repair Latency (ms) | Intercepted Violations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed_Framework** | `same_aisle` | 2.0 | 100.0% | 17.5 | +6.0% | 1.0 | 50.0% | 50.0% | 1190.5 ms | 0.0 |
| **Fixed_PIBT** | `same_aisle` | 2.0 | 100.0% | 20.0 | +0.0% | 0.0 | 0.0% | 0.0% | 0.2 ms | 0.0 |
| **Fixed_PBS** | `same_aisle` | 2.0 | 100.0% | 20.0 | +0.0% | 0.0 | 0.0% | 0.0% | 0.3 ms | 0.0 |
| **Ablation_NoCache** | `same_aisle` | 2.0 | 100.0% | 17.5 | +6.0% | 2.0 | 0.0% | 0.0% | 2317.7 ms | 0.0 |
| **Ablation_Thresh_070** | `same_aisle` | 2.0 | 100.0% | 17.5 | +6.0% | 1.0 | 50.0% | 50.0% | 1138.0 ms | 0.0 |
| **Ablation_Thresh_095** | `same_aisle` | 2.0 | 100.0% | 17.5 | +6.0% | 1.0 | 50.0% | 50.0% | 1246.3 ms | 0.0 |
