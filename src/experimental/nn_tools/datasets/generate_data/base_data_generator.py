# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
import logging
import numpy as np
import nd2py as nd
from abc import abstractmethod
from typing import Dict, List, Tuple
from sr_agent.utils import FactoryMixin

_logger = logging.getLogger(f'sr_agent.{__name__}')


class BaseDataGenerator(FactoryMixin):
    def __init__(self, sample_num=100, max_retry=5, random_seed=None, **kwargs):
        self.kwargs = kwargs
        self.max_retry = max(max_retry, 1)
        self.sample_num = sample_num
        self.random_seed = random_seed if random_seed is not None else np.random.randint(1e9)
        self._rng = np.random.default_rng(self.random_seed)

    def __call__(self, eq: nd.Symbol, _rng=None) -> Tuple[Dict[str, np.ndarray], np.ndarray, bool]:
        """ 调用 generate_data 接口生成数据，并计算目标值。
        如果生成的数据中包含无效值 (nan / inf)，则将无效部分重新生成，直到成功或达到 max_retry 次。
        返回 (data_dict, target, success), 其中 success 表示是否成功生成了足够的有效数据 (>= sample_num)
        如果 success 为 False，可以考虑提升 max_retry, 也可以考虑是 eq 的问题（例如没有有效定义域）
        """
        if _rng is not None:
            _rng_old, self._rng = self._rng, _rng
        try:
            valid_data_count = 0
            valid_data_dict = []
            variables = sorted({var.name for var in eq.iter_preorder() if isinstance(var, nd.Variable)})
            for _ in range(self.max_retry):
                data_dict = self.generate_data(eq, variables)
                target = self._eval_target(eq, data_dict)
                valid_idx = np.isfinite(target)
                valid_data_count += int(valid_idx.sum())
                valid_data_dict.append({var: data_dict[var][valid_idx] for var in variables})
                if valid_data_count >= self.sample_num:
                    break
            if failure := (valid_data_count < self.sample_num):
                _logger.warning(f"Only {valid_data_count} valid samples generated for equation {eq} after {self.max_retry} retries.")
                # 将一部分 invalid_data 也加入 valid_data_dict, 凑齐 sample_num 的要求
                needed = self.sample_num - valid_data_count
                invalid_idx = ~valid_idx
                add_idx = np.flatnonzero(invalid_idx)[:needed]
                valid_data_dict.append({var: data_dict[var][add_idx] for var in variables})
            valid_data_dict = {var: np.concatenate([d[var] for d in valid_data_dict], axis=0)[:self.sample_num] for var in variables}
            return valid_data_dict, self._eval_target(eq, valid_data_dict), not failure
        finally:
            if _rng is not None:
                self._rng = _rng_old

    @abstractmethod
    def generate_data(self, eq: nd.Symbol, variables: List[str]) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Subclasses must implement generate_data method")

    def _eval_target(self, eq: nd.Symbol, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        target = np.asarray(eq.eval(data_dict), dtype=np.float32)
        if target.ndim == 0:
            if data_dict:
                shape0 = sum(0 * data for data in data_dict.values()).shape
            else:
                shape0 = (self.sample_num,)
            target = np.full(shape0, target, dtype=np.float32)
        return target.reshape(-1)


@BaseDataGenerator.register("gaussian")
class GaussianDataGenerator(BaseDataGenerator):
    """对每个变量，随机生成一个均值和标准差，从而生成高斯分布的数据。
    (注: 当 max_retry > 1 时, 每次 retry 的数据可能采用不同的均值和标准差)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_range = self.kwargs.get("mean_range", (-10, 10))
        self.std_range  = self.kwargs.get("std_range", (1, 5))

    def generate_data(self, eq: nd.Symbol, variables: List[str]) -> Dict[str, np.ndarray]:
        data = {}
        for var in variables:
            mean = self._rng.uniform(*self.mean_range)
            std = self._rng.uniform(*self.std_range)
            data[var] = self._rng.normal(mean, std, self.sample_num)
        return data


@BaseDataGenerator.register("uniform")
class UniformDataGenerator(BaseDataGenerator):
    """对每个变量，在指定范围内随机生成均匀分布的数据。
    (注: 当 max_retry > 1 时, 每次 retry 的数据可能采用不同的分布范围)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.range = self.kwargs.get("range", (-10, 10))

    def generate_data(self, eq: nd.Symbol, variables: List[str]) -> Dict[str, np.ndarray]:
        data = {}
        for var in variables:
            data[var] = self._rng.uniform(*self.range, self.sample_num)
        return data


@BaseDataGenerator.register("gmm")
class GMMDataGenerator(BaseDataGenerator):
    """对每个变量，生成混合高斯分布的数据。
    (注: 当 max_retry > 1 时, 每次 retry 的数据可能采用不同的分布参数)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.component_range = self.kwargs.get("component_range", (1, 5))
        self.mean_range = self.kwargs.get("mean_range", (-10, 10))
        self.std_range  = self.kwargs.get("std_range", (1, 5))

    def generate_data(self, eq: nd.Symbol, variables: List[str]) -> Dict[str, np.ndarray]:
        data = {}
        for var in variables:
            component_num = self._rng.integers(*self.component_range)
            samples_per_component = int(np.ceil(self.sample_num / component_num))
            var_data = []
            for _ in range(component_num):
                mean = self._rng.uniform(*self.mean_range)
                std = self._rng.uniform(*self.std_range)
                var_data.append(self._rng.normal(mean, std, samples_per_component))
            data[var] = np.concatenate(var_data, axis=0)[:self.sample_num]
        return data
