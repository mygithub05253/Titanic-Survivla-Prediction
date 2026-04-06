# EDA Step 1: 기초 통계 분석 보고서

## 1. 개요

**프로젝트**: BDAI 타이타닉 생존 예측 공모전
**단계**: EDA Step 1 - 기초 통계 분석 (Basic Statistics)
**작성일**: 2026-04-06
**목표**: 데이터의 전체적인 구조와 기본 통계량 파악

## 2. 데이터 요약

| 항목 | 값 |
|------|-----|
| 전체 행 수 | 891 |
| 전체 열 수 | 15 |
| 메모리 사용량 | 278.9 KB |
| 수치형 컬럼 | 6개 (survived, pclass, age, sibsp, parch, fare) |
| 범주형 컬럼 | 7개 (sex, embarked, class, who, deck, embark_town, alive) |
| 불리언 컬럼 | 2개 (adult_male, alone) |

## 3. 컬럼 분류 및 역할

### 사용 가능 피처 (11개)
- **수치형**: pclass (객실 등급), age (나이), sibsp (형제/배우자), parch (부모/자녀), fare (운임)
- **범주형**: sex (성별), embarked (탑승항), who (man/woman/child), deck (갑판)
- **불리언**: adult_male (성인 남성), alone (혼자 탑승)

### 주의 컬럼
- **class**: pclass와 동일 정보 (중복)
- **embark_town**: embarked와 동일 정보 (중복)

### 금지 컬럼
- **alive**: survived와 100% 동일 (target leakage) - 절대 사용 금지

## 4. 결측치 현황

| 컬럼 | 결측 수 | 결측 비율 | 심각도 |
|------|---------|-----------|--------|
| deck | 688 | 77.2% | Red (위험) |
| age | 177 | 19.9% | Orange (주의) |
| embarked | 2 | 0.2% | Yellow (경미) |
| embark_town | 2 | 0.2% | Yellow (경미) |
| 나머지 11개 | 0 | 0.0% | Green (완전) |

## 5. 수치형 변수 기술통계

### age (나이)
- 평균: 29.7세, 중앙값: 28.0세
- 범위: 0.42 ~ 80세
- 왜도: 0.389 (약간 오른쪽 치우침)
- 결측 177건 (19.9%) - 대체 전략 필요

### fare (운임)
- 평균: 32.2, 중앙값: 14.5
- 범위: 0 ~ 512.3
- 왜도: 4.787 (극심한 오른쪽 치우침)
- 0원 티켓 15건 존재 - 조사 필요

### sibsp / parch
- 대부분 0 (혼자 탑승): sibsp 68.2%, parch 76.1%
- 높은 왜도: sibsp 3.695, parch 2.749

## 6. 범주형 변수 분포

| 변수 | 카테고리 | 비율 |
|------|---------|------|
| sex | male 64.8%, female 35.2% | 남성 편향 |
| embarked | S 72.3%, C 18.9%, Q 8.6% | Southampton 지배적 |
| who | man 60.3%, woman 30.4%, child 9.3% | 성인 남성 다수 |
| class | Third 55.1%, First 24.2%, Second 20.7% | 3등급 과반 |
| deck | C 6.6%, B 5.3%, D~G 합계 10.8% | 77.2% 결측 |

## 7. 핵심 발견사항

1. **데이터 불균형**: 생존율 38.4% (342명 생존 / 549명 사망)
2. **age 결측**: 19.9% 결측 - pclass+sex 그룹별 대체 전략 권장
3. **deck 결측**: 77.2% - 활용 가능하나 주의 필요
4. **fare 이상**: 극심한 왜도(4.79), log 변환 고려 필요
5. **fare 0원**: 15건 존재 - 이상치 또는 특수 케이스
6. **중복 컬럼 확인**: class=pclass, embark_town=embarked
7. **target leakage 확인**: alive=survived (사용 금지)

## 8. 다음 단계

**EDA Step 2: 타겟 클래스 불균형 분석**으로 진행
- 생존율 38.4%의 불균형이 F1 Score에 미치는 영향 분석
- 그룹별 생존율 비교
