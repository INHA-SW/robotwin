# RoboTwin ARM runtime source snapshot

This repository was recovered on 2026-08-15 from the server2 deployment at
`/data/seongwoong/git_repo/arm/benchmarks/robotwin`.

- The recovered host overlay did not contain Git metadata. A later read-only
  inspection of the validated `arm-openpi-robotwin-trace:20260827` image found
  a clean `/opt/robotwin` checkout at upstream commit
  `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`. The image identity was
  `sha256:91458c838a893561c5856ed78e389afe4f261b72f64a8172f66008f61d494b3e`.
  Treat that pair as the reproducible base for the current server2/Home
  deployment, not as proof that every older recovered run used the same base.
- Only executable source under `envs/`, `policy/`, and `script/` is preserved
  here. The 16 GB `assets/` tree remains a host-local runtime asset and is not
  part of this repository.
- The recovery archive SHA-256 is
  `a432c8ff2b62261e0573c5795dd987b690964921137a1b9064ee72a2638338b5`.
- The recovered adapters include the pi0.5 and LeRobot M1R1/RTC evaluation
  paths used by the 2026-08-13--14 RoboTwin experiments.
- Upstream reference: <https://github.com/robotwin-Platform/robotwin>.

Do not claim upstream-code identity from this snapshot. Rebase or transplant
the ARM-specific changes only after identifying the matching upstream commit.
