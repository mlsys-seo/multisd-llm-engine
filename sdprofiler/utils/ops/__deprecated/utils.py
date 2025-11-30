import math

def get_block_size(H: int) -> int:
    """
    1) 2의 제곱 중에서 H 이하이면서 가능한 한 큰 수를 선택
    2) 최소 32, 최대 1024 범위로 클램핑(clamp)
    """
    candidate = 2 ** int(math.log2(H)) if H > 0 else 1
    
    candidate = max(candidate, 32)
    candidate = min(candidate, 1024)
    return candidate
