# AGENTS.md

Questa guida definisce il workflow da seguire quando si modifica codice Python core.
Obiettivo: mantenere sempre allineati codice, guida tecnica e documentazione per file.

## Regola operativa principale

Quando viene modificato un file Python "core", nello stesso cambiamento devono essere aggiornati:

1. Il relativo file descrittivo `.tex` in `docs/tex/files/`.
2. Se necessario, il capitolo `.tex` in `docs/tex/chapters/` che lo include e lo contestualizza.
3. Se cambia il comportamento CLI/API, anche `README.md` o `PROJECT_STRUCTURE.md`.

## Mappa file core -> documentazione `.tex`

- `SIMULATOR/microgrid_simulator.py` -> `docs/tex/files/simulator_microgrid_simulator.tex`
- `SIMULATOR/tools.py` -> `docs/tex/files/simulator_tools.tex`
- `RULE_BASED/ems_offline.py` -> `docs/tex/files/rbc_ems_offline.tex`
- `RULE_BASED/RBC_EMS.py` -> `docs/tex/files/rbc_core_controller.tex`
- `MODEL_PREDICTIVE/ems_offline_mpc_v0.py` -> `docs/tex/files/mpc_ems_offline.tex`
- `MODEL_PREDICTIVE/mpc_MILP.py` -> `docs/tex/files/mpc_milp_core.tex`
- `RL_AGENT/ems_offline_RL_agent.py` -> `docs/tex/files/rl_runner.tex`
- `RL_AGENT/EMS_RL_agent.py` -> `docs/tex/files/rl_env_agent_core.tex`

## Cosa aggiornare nei `.tex` file

Ogni file `.tex` deve contenere almeno:

1. Scopo del file.
2. Interfacce principali (classi/funzioni/CLI argomenti).
3. Flusso dati input -> output.
4. Dipendenze critiche.
5. Effetti collaterali (I/O file, log, output, side effects).
6. Rischi e note di manutenzione.

## Validazione rapida prima del commit

Eseguire:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docs/check_code_docs_sync.ps1
```

Il controllo fallisce se un file Python core e cambiato ma il relativo `.tex` non e stato aggiornato.

## Quando aggiungere nuovi file alla mappa

Aggiungere la mappa quando:

- il file contiene logica di controllo (RBC/MPC/RL),
- oppure costruisce componenti del simulatore,
- oppure espone runner/entrypoint usati da script o esperimenti.

In quel caso:

1. Creare il nuovo `.tex` in `docs/tex/files/`.
2. Includerlo nel capitolo corretto in `docs/tex/chapters/`.
3. Aggiornare la mappa in questo `AGENTS.md`.
4. Aggiornare `scripts/docs/check_code_docs_sync.ps1`.
