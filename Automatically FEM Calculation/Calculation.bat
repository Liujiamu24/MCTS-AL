for %%g in (*.inp) do (
abaqus job=%%g cpus=16 int ask=off memory=32GB
)