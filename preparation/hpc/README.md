## Preparation with High Performance Computing (HPC)

- [x] Discuss the 5 annotated examples with Pascal
- [ ] Annote 100 abstracts with concepts: Progress 40/100
- [ ] Get access to weights directory (Felix)
- [ ] Fine-tune the model (Script)
- [ ] Generate concepts for 300 more abstracts
- [ ] Correct 300 concepts
- [ ] Fine-tune on these 300 abstracts

## Commands

`cd $(ws_find matconcepts)`

`sinfo_t_idle`

`sbatch --partition=dev_single --mem=96000 job.sh`

`scontrol show job <id>`
