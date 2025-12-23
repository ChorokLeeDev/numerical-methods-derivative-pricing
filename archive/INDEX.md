# 프로젝트 인덱스

금융공학 수치해석 코드베이스 가이드입니다.

## 디렉토리 구조

```
quant/
├── src/
│   ├── pricing/          # 옵션 가격결정 (해석해)
│   └── numerical/        # 수치해석 알고리즘
├── assignments/          # 과제 구현
├── docs/
│   ├── lectures/         # 강의자료
│   ├── references/       # 논문 및 이론
│   └── course/           # 과제 및 시험
└── tests/                # 단위 테스트
```

---

## 파일 가이드

### 옵션 가격결정 (`src/pricing/`)

| 파일 | 설명 | 언제 사용? |
|------|------|-----------|
| [blackscholes.py](src/pricing/blackscholes.py) | Black-Scholes 해석해 (유럽형 옵션) | 수치해법 벤치마크; 빠른 분석적 가격결정 |
| [ql_vanilla_option.py](src/pricing/ql_vanilla_option.py) | QuantLib 기반 유럽형 옵션 | 기간구조를 활용한 전문 가격결정 |
| [ql_worst_of.py](src/pricing/ql_worst_of.py) | Worst-of 바스켓 옵션 (Stulz 모델) | 두 기초자산 중 최소값 기준 옵션 |

### 수치해석 (`src/numerical/`)

| 파일 | 설명 | 언제 사용? |
|------|------|-----------|
| [fdm.py](src/numerical/fdm.py) | 유한차분법 (Explicit, Implicit, Crank-Nicolson) | PDE 기반 옵션 가격결정; 배리어 옵션 |
| [linear_system_direct.py](src/numerical/linear_system_direct.py) | 직접해법: LU, Cholesky, QR, SVD 분해 | 행렬 분해 학습; Ax=b 풀기 |
| [linear_system_iterative.py](src/numerical/linear_system_iterative.py) | 반복해법: Gauss-Seidel, SOR | 대형 희소행렬; 수렴 이해 |
| [linear_system_timing.py](src/numerical/linear_system_timing.py) | 선형 솔버 성능 벤치마크 | 알고리즘 효율성 비교 |
| [derivatives_error.py](src/numerical/derivatives_error.py) | 수치미분 근사 오차 분석 | 절단 오차 vs 반올림 오차 이해 |

### 과제 (`assignments/`)

| 파일 | 설명 | 언제 사용? |
|------|------|-----------|
| [fd_american_option.py](assignments/fd_american_option.py) | PSOR을 이용한 American 옵션 | 조기행사; 자유경계 문제 |
| [fd_worst_of.py](assignments/fd_worst_of.py) | OSM을 이용한 2D FDM | 다자산 PDE; 고급 수치해법 |

### 문서 (`docs/`)

**강의자료:**
- `lec01` - 수치해석 입문
- `lec02` - 선형방정식 시스템
- `lec03` - PDE를 위한 유한차분법
- `lec04` - 2요소 FDM (2차원 문제)

**참고문헌:**
- `stulz_1982_options_min_max.pdf` - Stulz (1982) worst-of 옵션 가격결정 논문
- `mean_value_theorem_taylor.pdf` - 수학적 기초
- `sor_matrix_derivation.pdf` - SOR 반복행렬 유도

**과제 자료:**
- `assignment.pdf` - 과제 문제
- `final_exam_2024.pdf` - 2024 기말고사
- `final_exam_2024_solutions.md` - 기말고사 해설
- `triangular_matrix_inverse.pdf` - 삼각행렬 역행렬

---

## 학습 경로

### 레벨 1: 기초
1. **derivatives_error.py** - 수치 근사 오차 이해
2. **linear_system_direct.py** - 행렬 분해 학습 (LU, Cholesky, QR)
3. **linear_system_iterative.py** - 반복 수렴 연구 (Gauss-Seidel, SOR)

### 레벨 2: 옵션 가격결정 기초
4. **blackscholes.py** - Black-Scholes 해석해 마스터
5. **ql_vanilla_option.py** - QuantLib으로 전문 가격결정
6. **fdm.py** - PDE용 유한차분법 구현

### 레벨 3: 고급 주제
7. **fd_american_option.py** - PSOR로 조기행사 처리
8. **ql_worst_of.py** - 다자산 파생상품 해석적 가격결정
9. **fd_worst_of.py** - Operator Splitting으로 2D PDE 해결

---

## 빠른 참조

### 파일별 핵심 개념

| 개념 | 파일 |
|------|------|
| Black-Scholes PDE | fdm.py, blackscholes.py |
| 행렬 분해 | linear_system_direct.py |
| 반복해법 | linear_system_iterative.py |
| 안정성 (CFL 조건) | fdm.py |
| American 옵션 | fd_american_option.py |
| 다자산 파생상품 | ql_worst_of.py, fd_worst_of.py |
| Greeks 계산 | fdm.py, fd_american_option.py |

### 의존성

- **QuantLib** - 전문 파생상품 라이브러리
- **NumPy** - 수치 배열
- **SciPy** - 선형대수, 보간
- **Pandas** - 데이터 구조
- **Matplotlib** - 시각화

---

## 테스트 실행

```bash
# 가상환경 활성화
source .venv/bin/activate

# 테스트 실행
pytest tests/ -v
```
