# big_wave_HDMI ㅡ Tienkung Pro에 대한 HDMI 학습 

HDMI 강화학습 프레임워크에 Tienkung Pro 휴머노이드 로봇을 적용한 저장소입니다.

전체 프로젝트의 세 번째 단계입니다.

1. [Kimodo](https://github.com/nv-tlabs/kimodo)               (Motion Generation)
 
2. [big_wave_GMR](https://github.com/cailab-hy/big_wave_GMR)     (Motion Retargeting)   
      
3. [big_wave_HDMI](https://github.com/cailab-hy/big_wave_HDMI)    (Policy Training) ← You are here
      
4. [big_wave_Deploy](https://github.com/cailab-hy/big_wave_Deploy)  (Real Robot Deployment)


## 목차

1. [요구 사항](#1-요구-사항)
2. [설치](#2-설치)
3. [학습](#3-학습)
4. [정책 재생 / 시각화 (play.py)](#4-정책-재생--시각화-playpy)
5. [평가 (eval.py)](#5-평가-evalpy)
6. [모션 데이터 확인](#6-모션-데이터-확인)
7. [SIM-TO-REAL](#7-sim-to-real)
8. [기타](#8-기타)
---

## 1. 요구 사항

### 1.1 검증된 환경

| 항목 | 버전 | 비고 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | |
| NVIDIA 드라이버 | 580.x (≥ 535) | `nvidia-smi`로 확인 |
| GPU | RTX 5000 Ada (16 GB) | VRAM에 맞춰 `task.num_envs` 조절 |
| Python | **3.11** | conda 환경 |
| Isaac Sim | **5.0.0** (pip) | `isaacsim[all,extscache]` |
| IsaacLab | **v2.3.0** | 소스 클론 + editable 설치 |
| PyTorch | **2.7.0+cu128** | Isaac Sim이 함께 설치 |
| NumPy | **1.26.x** | Isaac Sim 요구사항, 2.x 금지 |
| torchrl / tensordict | **0.7.0 / 0.7.0** | `setup.py`에 고정 |
| git-lfs | 3.x | `*.usd` 에셋이 LFS로 관리됨 |

### 1.2 사전 준비

```bash
# NVIDIA 드라이버 확인 (CUDA 12.8 런타임을 쓰므로 드라이버가 충분히 최신이어야 함)
nvidia-smi

# git-lfs 설치 (에셋 USD 파일 때문에 필수)
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install
```

---

## 2. 설치

전체 설치는 **conda 환경 → Isaac Sim → IsaacLab → 이 저장소** 순서입니다.
순서를 지켜야 합니다. `pip install -e .`가 `torch==2.7.0`을 고정하기 때문에, Isaac Sim보다 먼저 설치하면 CUDA 빌드가 어긋날 수 있습니다.

### 2.1 Conda 환경

```bash
conda create -n hdmi python=3.11 -y
conda activate hdmi
pip install --upgrade pip
```

### 2.2 Isaac Sim 5.0.0

```bash
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# 설치 확인 (GUI가 뜨면 성공, 첫 실행은 셰이더 캐시 때문에 몇 분 걸림)
isaacsim
```

### 2.3 IsaacLab v2.3.0

원하는 워크스페이스 아래에 클론합니다 (이 저장소와 형제 디렉터리일 필요는 없습니다).

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0

./isaaclab.sh -i none
```

### 2.4 이 저장소 (HDMI TienKung Pro)

```bash
git clone https://github.com/cailab-hy/big_wave_HDMI.git
cd big_wave_HDMI

# USD 에셋은 git-lfs로 저장되어 있음 (30개 파일). 반드시 받아야 함
git lfs pull

pip install -e .
```

### 2.5 설치 검증

```bash
conda activate hdmi
cd /path/to/big_wave_HDMI

# 1) 임포트 + CUDA 확인 → "2.7.0+cu128 True" 가 나와야 정상
python -c "import isaacsim, isaaclab, active_adaptation, torch; print(torch.__version__, torch.cuda.is_available())"

# 2) NumPy가 1.26.x인지 확인 (2.x면 Isaac Sim이 깨집니다)
python -c "import numpy; print(numpy.__version__)"

# 3) 레퍼런스 모션을 Isaac Sim에서 재생해 보기 (정책 없이 데이터만 확인)
python scripts/play.py algo=ppo_roa_train task=tienkung_pro/hdmi/scenario \
  +task.command.replay_motion=true headless=false task.num_envs=1
```

### 2.6 wandb 로그인 (선택)

학습 스크립트는 기본적으로 wandb에 로깅합니다(`wandb.mode: online`).

```bash
wandb login          # 로깅을 쓸 경우
# 또는 매 실행마다 wandb.mode=disabled 를 붙여 로컬로만 실행
```

### 2.7 실행 위치

**모든 스크립트는 저장소 루트에서 실행**해야 합니다 (`cfg/`, `data/` 경로가 상대 경로로 참조됩니다).

```bash
conda activate hdmi
cd /path/to/big_wave_HDMI
```

---

## 3. 학습

```bash
# teacher 학습 student 파인튜닝
python scripts/train_sequential.py  stages=[ppo_roa_train,ppo_roa_adapt,ppo_roa_finetune] task=tienkung_pro/hdmi/scenario task.num_envs=4096 

# 이전 스테이지의 wandb run 을 다음 스테이지 checkpoint 로 자동 연결)
python scripts/train_sequential.py task=tienkung_pro/hdmi/scenario  stages=[ppo_roa_train_est,ppo_roa_adapt_est] task.num_envs=2048 app.enable_cameras=true
```

---

## 4. 정책 재생 / 시각화 (`play.py`)

```bash
python scripts/play.py algo=ppo_roa_adapt_est task=tienkung_pro/hdmi/scenario \
  checkpoint_path=/abs/path/checkpoint_final.pt \
  headless=false task.num_envs=1 app.enable_cameras=true export_policy=true
```

#### 주의사항 

```export_policy=true``` 를 넣어야 [SIM-TO-REAL](#7-sim-to-real) 를 위한 정책 파일 (.onnx)이 생성됩니다.
생성된 onnx 파일은 "checkpoint_path" 에 저장됩니다.

e.g. ~/exports/TienkungProScenario/...


---

## 5. 평가 (`eval.py`)

정해진 개수의 **에피소드가 실제로 끝날 때까지** 롤아웃하면서 트래킹 오차와 성공률을 집계합니다. 학습은 하지 않고, 행동은 결정론적(`ExplorationType.MODE`)입니다.

```bash
python scripts/eval.py --task tienkung_pro/hdmi/scenario \
  --num_envs 200 --num_episodes 1000 --headless \
  --algo ppo_roa_adapt_est \
  --checkpoint /abs/path/checkpoint_final.pt
```

---

## 6. 모션 데이터 확인

모션은 `data/motion/<robot>/<task>/`에 `motion.npz` + `meta.json` 형태로 들어갑니다
(현재: `data/motion/tienkung_pro/scenario/`. 포맷은 아래 [Data Preparation](#data-preparation) 참조).

```bash
# Isaac Sim에서 레퍼런스 모션 재생
python scripts/play.py algo=ppo_roa_train task=tienkung_pro/hdmi/scenario \
  +task.command.replay_motion=true headless=false task.num_envs=1
```

---

## 7. sim-to-real 

해당 package를 실제 환경에서 돌리기 위해 아래 명시된 링크를 따라가세요.

[big_wave_deploy](https://github.com/cailab-hy/big_wave_Deploy)

---

## 8. 기타

학습, 시각화, 평가에 대한 자세한 사항은 문서를 참고하세요 

---

## Citation

If you use this code in your resarch, please cite this paper

```
@misc{weng2025hdmilearninginteractivehumanoid,
title={HDMI: Learning Interactive Humanoid Whole-Body Control from HUman Videos},
author={Haoyang Weng and Yitang Li and Nikhil Sobanbabu and Zihan Wang and Zhengyi Luo and Tairan He and Deva Ramanan and Guanya Shi},
year={2025},
eprint={2509.16757},
archivePrefix={arXiv},
primaryClass={cs.RO},
url={https://arxiv.org/abs/2509.16757},
}
```

---

본 프로젝트는 빅웨이브와 한양대학교 박태준 교수님 연구실 CAILAB에서 실시한 산학과제 결과물입니다.
