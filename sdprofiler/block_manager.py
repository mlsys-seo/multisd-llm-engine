from typing import List
from collections import deque

import numpy as np

from .registry import EngineRegistry
from .request import Request
from .utils.common import print_debug

DEBUG = False

class PagedAttentionBlockManager:
    def __init__(self, 
            registry: EngineRegistry,            
        ):
        self.registry = registry
        
        self.kv_cache_block_size = self.registry.get('val.engine.kv_cache_block_size')

        self.world_size = self.registry.get('val.model.world_size')
        self.use_data_parallel_draft = self.registry.get('val.engine.use_data_parallel_draft')
        if self.use_data_parallel_draft:
            self.num_kv_cache_blocks = self.registry.get('val.engine.num_kv_cache_blocks') // self.world_size
        else:
            self.num_kv_cache_blocks = self.registry.get('val.engine.num_kv_cache_blocks')


        self.free_blocks = deque(list(range(self.num_kv_cache_blocks)), maxlen=self.num_kv_cache_blocks)


    @property
    def num_free_blocks(self):
        return len(self.free_blocks)
    
    @property
    def num_using_blocks(self):
        return self.num_kv_cache_blocks - self.num_free_blocks
    
    def num_blocks_needed(
            self,
            num_tokens: int
        ) -> int:
        return (num_tokens + self.kv_cache_block_size - 1) // self.kv_cache_block_size

    def check_if_blocks_are_available(
            self,
            request: Request
        ) -> bool:
        tokens_needed = request.get_required_num_tokens(self.kv_cache_block_size)
        blocks_needed = self.num_blocks_needed(tokens_needed)

        return len(self.free_blocks) >= blocks_needed
        
    def allocate_blocks_on_request(
            self,
            request: Request
        ) -> List[int]:
        if DEBUG:
            print_debug(
                function_name="PagedAttentionBlockManager.allocate_blocks_on_request",
                request_kv_blocks_needed=request.kv_blocks_needed,
                num_free_blocks=self.num_free_blocks,
                num_using_blocks=self.num_using_blocks,
            )
        allocated_blocks = np.array([self.free_blocks.popleft() for _ in range(request.kv_blocks_needed)], dtype=np.int32)
        request.append_paged_kv_blocks(allocated_blocks)
        return request
    
                
    def deallocate_blocks_on_request(
            self, 
            request: Request
        ):
        self.free_blocks.extendleft(request.paged_kv_cache.paged_kv_block_indices)
        