# -*- coding: utf-8 -*-
"""minivllm 单元测试：分词 / 采样 / 分页缓存 / 引擎 端到端。"""
import torch

from minivllm import (MiniBPETokenizer, Sampler, BlockAllocator,
                      PagedKVCache, TinyGPT, LLMEngine)


# ---------- tokenizer ----------
def test_bpe_roundtrip():
    tok = MiniBPETokenizer(vocab_size=300).train(
        ["the quick brown fox " * 10, "large language model " * 10])
    text = "the quick large model"
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == 300
    assert tok.compression_ratio("the quick brown fox the quick brown fox") > 1


# ---------- sampler ----------
def test_greedy_matches_argmax():
    s = Sampler(greedy=True)
    logits = torch.tensor([[0.1, 3.0, 0.2]])
    assert int(s(logits)) == 1


def test_temperature_zero_is_greedy():
    s = Sampler(temperature=0.0)
    logits = torch.tensor([[0.1, 3.0, 0.2]])
    assert int(s(logits)) == 1


def test_topk_never_picks_outside_k():
    s = Sampler(temperature=1.0, top_k=2, seed=0)
    logits = torch.tensor([[0.0, 0.0, 5.0, 4.5, -10.0]]).repeat(50, 1)
    picks = {int(x) for x in s(logits)}
    assert picks <= {2, 3}


def test_topp_excludes_tail():
    s = Sampler(temperature=1.0, top_p=0.5, seed=0)
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]]).repeat(50, 1)
    picks = {int(x) for x in s(logits)}
    assert picks == {0}


# ---------- paged kv ----------
def test_allocator_refcount():
    al = BlockAllocator(num_blocks=4, block_size=4)
    a = al.allocate(); al.retain(a)
    al.release(a)
    assert al.num_free == 3            # 还有一份引用
    al.release(a)
    assert al.num_free == 4            # 引用归零才回收


def test_paged_kv_write_gather_roundtrip():
    cache = PagedKVCache(num_blocks=8, block_size=4, n_heads=2, head_dim=3, n_layers=1)
    cache.new_seq(0)
    for t in range(9):                 # 跨 3 个物理块
        k = torch.full((2, 3), float(t))
        cache.write(0, 0, k, k + 1)
    K, V = cache.gather(0, 0)
    assert K.shape == (9, 2, 3)
    assert torch.equal(K[8], torch.full((2, 3), 8.0))
    assert torch.equal(V[8], torch.full((2, 3), 9.0))
    assert cache.fragmentation() > 0   # 最后一块没写满 → 有内部碎片


def test_fork_shares_blocks():
    cache = PagedKVCache(num_blocks=8, block_size=4, n_heads=1, head_dim=2, n_layers=1)
    cache.new_seq(0)
    for _ in range(5):
        cache.write(0, 0, torch.ones(1, 2), torch.ones(1, 2))
    assert cache.allocator.num_free == 6   # 5 个 token 占 2 块
    cache.fork(0, 1)                   # 序列 1 共享序列 0 的块（refcount=2）
    assert cache.allocator.num_free == 6
    cache.drop_seq(0)
    assert cache.allocator.num_free == 6   # 还有 seq1 持有引用，不回收
    cache.drop_seq(1)
    assert cache.allocator.num_free == 8   # 引用归零才回收



# ---------- engine 端到端 ----------
def _make_engine(max_running=2, num_blocks=32):
    tok = MiniBPETokenizer(vocab_size=280).train(["the quick brown fox " * 8])
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2)
    cache = PagedKVCache(num_blocks=num_blocks, block_size=8,
                         n_heads=model.n_heads, head_dim=model.head_dim,
                         n_layers=model.n_layers)
    engine = LLMEngine(model, cache, Sampler(temperature=0.9, top_k=10, seed=0),
                       max_running=max_running)
    return engine


def test_engine_runs_and_finishes():
    eng = _make_engine()
    for _ in range(3):
        eng.add_request([1, 2, 3], max_new_tokens=8)
    outs = eng.run()
    assert len(outs) == 3
    for ids in outs.values():
        assert 1 <= len(ids) <= 8


def test_engine_concurrent():
    eng = _make_engine(max_running=2)
    for _ in range(4):
        eng.add_request([1, 2, 3, 4], max_new_tokens=6)
    outs = eng.run()
    assert len(outs) == 4
    assert all(1 <= len(v) <= 6 for v in outs.values())
