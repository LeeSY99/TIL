https://www.notion.so/Attention-Is-All-You-Need-2dfd9b92b2c78048ba06c3cc46ba9e16?source=copy_link
1. **배경**
    - 시퀀스 모델링 :  시퀀스를 가지는 데이터로부터 또 다른 시퀀스를 생성하는 작업 (기계번역, 챗봇 등)
    - RNN, LSTM, GRU는 시퀀스 모델링 및 변환 문제에서 가장 sota한 접근법으로 자리잡고 있음
    - 이런 순차적인 특성 때문에 하나의 학습 예제 내부에서 병렬화가 어렵고 메모리와 연산에서 많은 부담. 그리고 긴 시퀀스에서 의존성 학습이 힘든 문제점 존재
2. **Seq2Seq**
    - 인코더의 출력을 context vector로써 디코더의 입력으로 들어감
    - 메모리의 한계로 인해 이 벡터의 크기를 제한(고정)
    - 모든 정보가 이 안에 들어갈 수가 없음. 길수록 정보의 손실이 커지고 정보를 압출할 때 병목현상이 일어남.
    
    → 분해factorization 기법과 조건부 계산(conditional computation)을 통해 계산 효율을 크게 향상시켰고 모델 성능도 향상 시켰지만 순차적 계산이라는 근본적인 제약은 남아있음
    
3. **Transformer의 등장**
    - Attention Mechanism: 이전 히든스테이트만 고려하지 않고 전체 시퀀스의 구조를 파악하여 메모리 효율과 최고의 성능
4. **구조**
    - 기존 좋은 성능을 보이던 인코더 디코더 구조는 동일
    - 내부는 self-attention과 fully connected layer으로만 구성
    - 입력: Input Embedding, Positional Encoding을 거쳐 입력
    - 인코더
        - 토큰 → 임베딩
        - 위치 인코딩 더하기 (순서 정보)
        - Multi-Head Encoder Self-Attention (여러개. 논문에선 6개)
        - Feed-Forward
    - 디코더
        - Masked Multi-Head Self-Attention
            - 디코더 내부 토큰끼리 self-attention 하지만 미래 토큰을 못 보게 마스킹
            - Encoder-Decoder Attention(cross Attention)
            - Feed-Forward
5. **Attention**
    - 쿼리와 키-값(key-value) 쌍들의 집합을 받아서 출력을 만듦. 출력은 값(value)의 가중합. 가중치는 쿼리와 해당 키의 적합도로 계산
    - 쉽게 말하면 같은 문장 내에서 단어들 간의 관계
        - 찾고 싶은 것 : Q (Query)
        - 참고할 후보들 점수 매기기 : Q와 K(Key)비교해서 점수
        - 내용 가져와 섞기: 점수(가중치)로 V(Value)를 섞어 결과 만들기
        
        → Q, K의 유사한 만큼 V를 가져온다.
        
        예시
        
        - 철수는 사과를 먹었다. 그는 배가 고팠다.
        - 그는 ← 이 누구인지 이해하려면 “철수”를 더 많이 참고, “사과”는 덜 참고
    - **Scaled Dot-Product Attention**
        
        ![image.png](attachment:117c6412-343e-4ebf-986f-74a424c493b9:image.png)
        
        - 모델 차원: 512, 헤드 수: 8 → Q/K/V: 64차원
        - **쿼리와 모든 키의 dot-product를 계산하고 루트 d_k로 나눈 다음 softmax 적용해 값들에 대한 가중치 얻음 → 가중치로 값 V를 가중합해 출력**
        - 루트d_k[key(query) 벡터의 차원] 로 나누는 이유는 d_k 값이 너무 커지면 행렬 연산 값도 커져 softmax가 포화되기 쉬우니 스케일을 안정화.
    - **Multi-Head Attention**
        - 어텐션을 한번만 하지 말고 서로 다른 관점에서 여러번 동시에 해서 합치자
        - 왜 Head를 여려개로 했을까
            - 단일 head는 한 개의 가중치 표(softmax 결과)로 모든 정보를 섞어야 함 → 정보가 흐릿해질 수 있음
            - head마다 다른 관점으로 보고 여러 관계를 동시에 학습 후 합침
            - 연산 비용은 같다(Head 당 차원을 줄였기 때문)
6. Transformer에서 Attention
    1. **Encoder Self-Attention**
        - Q/K/V : 전부 인코더의 입력 (같은 시퀀스)
        - 각 토큰 위치는 현재 토큰이 문장 전체의 토큰을 얼마나 참고할지 계산해서 그 결과로 새 토큰 표현을 만듦 (Attention은 가중치가 목적이 아니라, V를 섞어서 **새 토큰 벡터**를 만드는 게 목적)
        - 인코더의 레이어를 통과하면서 이전 레이어의 표현을 입력으로 받아 각 토큰이 다른 모든 토큰을 다시 참고해 표현 업데이트
    2. **Decoder Masked Self-Attention**
        - Q/K/V : 디코더의 현재 입력 Y (현재까지 생성된 토큰)
            - 무슨말인가?하면
            - 학습할 때
                - 정답(타깃): `y = [y1, y2, y3, ...]`
                - 디코더 입력: `[<BOS>, y1, y2, y3, ...]` 한 칸 shift함
                - y_i를 맞추기 위해 y_i-1까지만 갖고 있는 구조
            - 추론:
                - 생성됨: `[y1, y2]`
                - 다음 토큰 y3을 만들때 디코더의 입력 `[<BOS>, y1, y2]`
        - 마스크: 미래 토큰을 못 보게 하는 것
    3. Cross Attention, Encoder–Decoder Attention
        - Q : 디코더 이전 레이어 출력
        - K/V : 인코더 출력
        - 디코더의 시퀀스들이 인코더의 시퀀스들과의 연관성을 학습할 수 있다
7. Positional Encoding
    - 시간의 연속성을 다루지 않기 때문에 sin,cos 함수를 이용해 상대적인 위치를 더해줌
8. Why Self-Attention?
    1. **Total computational complexity per layer**
        - 문장 길이가 차원보다 작으면 Self-Attention이 복잡도가 낮아짐
        - 보통 문장 길이가 256 또는 512를 넘기는 경우가 잘 없어 효율적
        - n이 커져도 근처만 보는 attention도 있다고 언급
    2. **The amount of computation**
        - RNN: 순차적으로 n개의 시퀀스 만큼 입력 받음 → O(n)
        - Self-Attention: 모든 포지션들의 어텐션 스코어를 한번에 처리 → 대략 O(1) → 병렬화 유리
    3. **Path length between long-range dependencies**
        - **Self-attention**: 어떤 두 위치든 한 번의 attention으로 직접 연결 → 경로 길이 **O(1)**
        - **RNN**: 정보가 step을 따라 전달 → 멀리 갈수록 경로가 길어짐(대략 O(n))
        - **CNN**: 커널 폭 k면 한 층에서 k 근처만 연결 → 멀리 연결하려면 여러 층 필요
        
        경로가 짧을수록(backprop 포함) 장거리 의존성 학습이 쉬워진다
