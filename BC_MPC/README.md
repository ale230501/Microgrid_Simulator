# Behavior Cloning (BC) da MPC

Questa cartella contiene uno **scheletro operativo** per generare un dataset di dimostrazioni con l'MPC
(`MODEL_PREDICTIVE/mpc_MILP.py`) e addestrare un agente via **Behavior Cloning** usando la libreria
`imitation`.

## Quando ha senso farlo
- Se l'MPC produce traiettorie **stabili** e coerenti (anche se non ottime), il BC può
  replicare quel comportamento in modo più veloce/inferenza più leggera.
- Il BC **non supera** l'MPC da cui impara: se l'MPC è subottimo, anche il BC lo sarà.
- Il BC richiede coerenza tra **spazio delle osservazioni** e **spazio delle azioni**
  (normalizzazione inclusa): i file di dataset e le configurazioni devono essere allineati.

## Flusso consigliato
1. **Genera dataset di dimostrazioni** dall'MPC (stato → azione).
2. **Allena il BC** con `imitation`.
3. **(Opzionale)** Training con PPO per migliorare la policy BC.
4. **Valuta** il BC/ppo trained nello stesso ambiente offline usato dall'RL.

## Script disponibili
- `generate_mpc_bc_dataset.py`: raccoglie osservazioni e azioni dell'MPC in un file `.npz`.
- `train_bc_from_mpc.py`: allena un agente BC su quel dataset.

## Esempio rapido
```bash
python BC_MPC/generate_mpc_bc_dataset.py \
  --rl-config configs/controllers/rl/opsd/params_RL_agent_OPSD_1_DAY.yml \
  --mpc-config configs/controllers/mpc/params_OPSD.yml \
  --dataset-path data/DE_KN_residential1_train.csv \
  --output-dir BC_MPC/outputs

python BC_MPC/train_bc_from_mpc.py \
  --rl-config configs/controllers/rl/opsd/params_RL_agent_OPSD_1_DAY.yml \
  --dataset BC_MPC/outputs/mpc_bc_dataset.npz \
  --output-dir BC_MPC/outputs/bc_policies

python BC_MPC/training_rl_agent_ppo.py \
  --rl-config configs/controllers/rl/opsd/params_RL_agent_OPSD_1_DAY.yml \
  --bc-policy BC_MPC/outputs/bc_policies/<run>/bc_policy.pt \
  --output-dir BC_MPC/outputs/ppo_trained \
  --policy-net 64,64 \
  --timesteps 200000
```

## Note importanti
- **Normalizzazione**: se in `rl.normalization.actions.enabled` è `true`, il dataset
  salva azioni già normalizzate; il BC imparerà in quello spazio.
- **Lunghezze**: il dataset viene limitato alla lunghezza massima compatibile con
  le serie storiche e con l'orizzonte MPC.
- **Configurazioni**: il file RL (in `configs/controllers/rl/...`) e quello MPC (in `configs/controllers/mpc/...`)
  devono riferirsi allo **stesso dataset** per evitare mismatch fra osservazioni e forecast.
- **Policy net**: il training PPO richiede un'architettura compatibile con la policy BC
  (di default `FeedForward32Policy`, quindi `32,32`). Se cambi la rete BC, usa lo stesso
  `--policy-net` nel fine-tuning.
- **Output**: di default BC e PPO salvano in `BC_MPC/outputs/bc_policies/<run>` e
  `BC_MPC/outputs/ppo_trained/<run>`, includendo i params usati.
