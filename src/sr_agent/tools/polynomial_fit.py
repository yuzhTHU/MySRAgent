# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""多项式拟合工具。

提供对输入数据的多项式拟合功能，支持自定义最高阶次数、交叉项控制等。
"""

import numpy as np
from itertools import combinations, product
from typing import Dict, Any, List, Optional, Tuple, Set

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('polynomial_fit')
class PolynomialFitTool(BaseTool):
    """对数据进行多项式拟合。

    本工具使用最小二乘法对输入数据进行多项式拟合，支持：
    - 自定义最高阶次数
    - 是否包含交叉项
    - 交叉项黑名单（哪些变量之间不产生交叉项）
    - 交叉项白名单（只允许哪些变量之间产生交叉项）

    返回结果包括：
    - 拟合的多项式表达式
    - 各系数的值及其统计显著性
    - 整体拟合效果（R²、调整 R²、RMSE 等）

    适用场景：
    - 探索变量间的非线性关系
    - 符号回归的预处理或基线
    - 特征工程中的多项式特征生成
    """

    metadata = ToolMetadata(
        name="polynomial_fit",
        description="对数据进行多项式拟合，支持自定义阶数、交叉项控制。返回拟合的多项式及其统计显著性、拟合效果指标。",
        category="regression",
    )

    def execute(
        self,
        x_vars: Optional[List[str]] = None,
        y_var: str = "y",
        max_degree: int = 2,
        include_interactions: bool = True,
        interaction_blacklist: Optional[List[Tuple[str, str]]] = None,
        interaction_whitelist: Optional[List[Tuple[str, str]]] = None,
        include_bias: bool = True,
    ) -> Dict[str, Any]:
        """执行多项式拟合。

        Args:
            x_vars: 输入特征名列表，如 ["x1", "x2"]。None 表示使用全部特征。
            y_var: 目标变量名，默认为 "y"。
            max_degree: 多项式最高阶次数，默认为 2。
            include_interactions: 是否包含交叉项，默认为 True。
            interaction_blacklist: 交叉项黑名单，指定哪些变量之间不产生交叉项。
                例如 [("x1", "x2")] 表示 x1 和 x2 之间不产生交叉项。
            interaction_whitelist: 交叉项白名单，只允许指定的变量对产生交叉项。
                如果为 None，则允许所有变量对（除非在黑名单中）。
                如果指定，则只生成白名单中的交叉项。
            include_bias: 是否包含截距项，默认为 True。

        Returns:
            包含以下字段的字典：
            - polynomial: 多项式字符串表达式
            - terms: 各多项式项的详细信息列表
            - coefficients: 系数字典 {term: coefficient}
            - fit_quality: 拟合质量指标（R²、调整 R²、RMSE、AIC、BIC）
            - design_matrix_shape: 设计矩阵的形状
            - n_parameters: 参数数量
            - warnings: 警告信息列表
        """
        x = self.context['x']
        y = self.context['y']

        # 选择要使用的变量
        if x_vars is None:
            x_vars = list(x.keys())

        # 从 context 中获取数据
        x = {var: x[var] for var in x_vars}

        warnings = []

        # 数据验证
        y = np.asarray(y).flatten()
        n_samples = len(y)

        var_names = list(x.keys())
        n_features = len(var_names)

        if n_features == 0:
            raise ValueError("至少需要一个输入特征")

        # 确保所有特征长度一致
        for name, arr in x.items():
            arr_checked = np.asarray(arr).flatten()
            if len(arr_checked) != n_samples:
                raise ValueError(f"特征 {name} 的长度 ({len(arr_checked)}) 与 y 的长度 ({n_samples}) 不一致")
            x[name] = arr_checked

        # 生成交叉项限制
        allowed_interactions = self._get_allowed_interactions(
            var_names,
            include_interactions,
            interaction_blacklist,
            interaction_whitelist
        )

        # 生成所有多项式项的幂次组合
        term_powers = self._generate_term_powers(
            var_names,
            max_degree,
            allowed_interactions
        )

        # 构建设计矩阵和项名称
        design_matrix, term_info = self._build_design_matrix(
            x, var_names, term_powers, include_bias
        )

        # 检查设计矩阵的秩
        matrix_rank = np.linalg.matrix_rank(design_matrix)
        n_params = design_matrix.shape[1]

        if matrix_rank < n_params:
            warnings.append(
                f"设计矩阵秩 deficient: 秩={matrix_rank}, 参数={n_params}。"
                "可能存在多重共线性，结果可能不稳定。"
            )

        # 使用最小二乘法拟合
        n = n_samples
        p = n_params

        try:
            # 使用 QR 分解提高数值稳定性
            Q, R = np.linalg.qr(design_matrix)
            coefficients = np.linalg.solve(R, Q.T @ y)

            # 计算残差
            y_pred = design_matrix @ coefficients
            residuals = y - y_pred

            # 计算系数标准误差
            if n > p:
                mse = np.sum(residuals ** 2) / (n - p)
                # 系数的协方差矩阵
                try:
                    cov_matrix = mse * np.linalg.inv(R.T @ R)
                    std_errors = np.sqrt(np.diag(cov_matrix))
                except np.linalg.LinAlgError:
                    # 如果矩阵奇异，使用伪逆
                    cov_matrix = mse * np.linalg.pinv(R.T @ R)
                    std_errors = np.sqrt(np.diag(cov_matrix))
                    warnings.append("使用伪逆计算标准误差，结果可能不够精确。")
            else:
                std_errors = np.full(n_params, np.nan)
                warnings.append("样本数不足以计算标准误差。")

            # 计算 t 统计量和 p 值
            with np.errstate(divide='ignore', invalid='ignore'):
                t_stats = coefficients / std_errors
                # 使用 t 分布计算双尾 p 值
                from scipy import stats
                dof = max(n - p, 1)
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

        except Exception as e:
            # 降级到普通最小二乘
            warnings.append(f"QR 分解失败，使用普通最小二乘法：{str(e)}")
            coefficients, residuals, rank, s = np.linalg.lstsq(
                design_matrix, y, rcond=None
            )
            y_pred = design_matrix @ coefficients
            std_errors = np.full(n_params, np.nan)
            t_stats = np.full(n_params, np.nan)
            p_values = np.full(n_params, np.nan)

        # 计算拟合质量指标
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 调整 R²
        if n > p and ss_tot > 0:
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
        else:
            adj_r_squared = np.nan

        # RMSE 和 MAE
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        # AIC 和 BIC
        if n > p and ss_res > 0:
            log_likelihood = -n/2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
            aic = 2 * p - 2 * log_likelihood
            bic = p * np.log(n) - 2 * log_likelihood
        else:
            aic = np.nan
            bic = np.nan

        # 构建结果
        terms_result = []
        polynomial_parts = []
        coefficients_dict = {}

        for i, (term_str, coef, std_err, t_stat, p_val) in enumerate(
            zip(term_info, coefficients, std_errors, t_stats, p_values)
        ):
            term_result = {
                "term": term_str,
                "coefficient": float(coef),
                "std_error": float(std_err) if not np.isnan(std_err) else None,
                "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                "p_value": float(p_val) if not np.isnan(p_val) else None,
                "significant_at_0.05": bool(p_val < 0.05) if not np.isnan(p_val) else None,
                "significant_at_0.01": bool(p_val < 0.01) if not np.isnan(p_val) else None,
            }
            terms_result.append(term_result)
            coefficients_dict[term_str] = float(coef)

            # 构建多项式字符串部分
            if term_str == "1":
                polynomial_parts.insert(0, f"{coef:.6f}")
            elif coef != 0:
                if coef < 0:
                    sign = "-"
                    abs_coef = abs(coef)
                else:
                    sign = "+" if polynomial_parts else ""
                    abs_coef = coef

                if abs_coef == 1:
                    term_display = f"{sign} {term_str}"
                else:
                    term_display = f"{sign} {abs_coef:.6f}*{term_str}"
                polynomial_parts.append(term_display.strip())

        # 构建多项式字符串
        if not polynomial_parts:
            polynomial_str = "0"
        else:
            polynomial_str = " ".join(polynomial_parts)
            # 处理开头的 "+"
            if polynomial_str.startswith("+"):
                polynomial_str = polynomial_str[1:].strip()

        return {
            "polynomial": polynomial_str,
            "terms": terms_result,
            "coefficients": coefficients_dict,
            "fit_quality": {
                "r_squared": float(r_squared),
                "adjusted_r_squared": float(adj_r_squared) if not np.isnan(adj_r_squared) else None,
                "rmse": float(rmse),
                "mae": float(mae),
                "aic": float(aic) if not np.isnan(aic) else None,
                "bic": float(bic) if not np.isnan(bic) else None,
            },
            "residuals": residuals.tolist(),
            "predictions": y_pred.tolist(),
            "design_matrix_shape": design_matrix.shape,
            "n_parameters": n_params,
            "n_samples": n_samples,
            "matrix_rank": matrix_rank,
            "warnings": warnings,
        }

    def _get_allowed_interactions(
        self,
        var_names: List[str],
        include_interactions: bool,
        blacklist: Optional[List[Tuple[str, str]]],
        whitelist: Optional[List[Tuple[str, str]]],
    ) -> Set[Tuple[str, str]]:
        """获取允许的交叉项组合。

        Args:
            var_names: 变量名列表。
            include_interactions: 是否包含交叉项。
            blacklist: 交叉项黑名单。
            whitelist: 交叉项白名单。

        Returns:
            允许的交叉项组合集合。
        """
        if not include_interactions:
            return set()

        # 生成所有可能的变量对
        all_pairs = set(combinations(sorted(var_names), 2))

        if whitelist is not None:
            # 白名单模式：只允许白名单中的组合
            whitelist_normalized = set(
                tuple(sorted(pair)) for pair in whitelist
            )
            allowed = all_pairs & whitelist_normalized
        else:
            # 默认允许所有组合，除非在黑名单中
            allowed = all_pairs

        if blacklist is not None:
            blacklist_normalized = set(
                tuple(sorted(pair)) for pair in blacklist
            )
            allowed -= blacklist_normalized

        return allowed

    def _generate_term_powers(
        self,
        var_names: List[str],
        max_degree: int,
        allowed_interactions: Set[Tuple[str, str]],
    ) -> List[Tuple[int, ...]]:
        """生成所有多项式项的幂次组合。

        使用递归方式生成所有满足总次数 <= max_degree 的幂次组合。

        Args:
            var_names: 变量名列表。
            max_degree: 最高阶次数。
            allowed_interactions: 允许的交叉项组合。

        Returns:
            幂次组合列表，每个组合是一个元组，表示各变量的幂次。
        """
        n_vars = len(var_names)
        term_powers = []

        # 生成所有可能的幂次组合（使用 product 生成笛卡尔积）
        for powers in product(range(max_degree + 1), repeat=n_vars):
            total_degree = sum(powers)
            if total_degree == 0:
                # 截距项，总是添加
                term_powers.append(powers)
            elif total_degree <= max_degree:
                # 检查是否满足交叉项约束
                if self._check_interaction_constraint(
                    powers, var_names, allowed_interactions
                ):
                    term_powers.append(powers)

        # 按总次数排序，确保截距项在第一
        term_powers.sort(key=lambda p: (sum(p), p))

        return term_powers

    def _check_interaction_constraint(
        self,
        powers: Tuple[int, ...],
        var_names: List[str],
        allowed_interactions: Set[Tuple[str, str]],
    ) -> bool:
        """检查幂次组合是否满足交叉项约束。

        Args:
            powers: 幂次元组。
            var_names: 变量名列表。
            allowed_interactions: 允许的交叉项组合。

        Returns:
            是否满足约束。
        """
        if not allowed_interactions:
            # 不允许任何交叉项，检查是否有多于一个变量非零
            non_zero_count = sum(1 for p in powers if p > 0)
            return non_zero_count <= 1

        # 找出所有同时非零的变量对
        non_zero_indices = [i for i, p in enumerate(powers) if p > 0]

        for i, j in combinations(non_zero_indices, 2):
            pair = tuple(sorted((var_names[i], var_names[j])))
            if pair not in allowed_interactions:
                return False

        return True

    def _build_design_matrix(
        self,
        x: Dict[str, np.ndarray],
        var_names: List[str],
        term_powers: List[Tuple[int, ...]],
        include_bias: bool,
    ) -> Tuple[np.ndarray, List[str]]:
        """构建设计矩阵。

        Args:
            x: 输入特征字典。
            var_names: 变量名列表。
            term_powers: 幂次组合列表。
            include_bias: 是否包含截距项。

        Returns:
            设计矩阵和项名称列表。
        """
        n_samples = len(next(iter(x.values())))
        term_info = []
        columns = []

        # 添加所有多项式项
        for powers in term_powers:
            # 跳过截距项（如果不需要）
            if all(p == 0 for p in powers):
                if not include_bias:
                    continue
                # 添加截距项
                term_info.append("1")
                columns.append(np.ones(n_samples))
                continue

            # 计算当前项的值
            term_values = np.ones(n_samples)
            for var_idx, power in enumerate(powers):
                if power > 0:
                    var_name = var_names[var_idx]
                    term_values *= x[var_name] ** power

            columns.append(term_values)
            term_info.append(self._powers_to_string(powers, var_names))

        design_matrix = np.column_stack(columns) if columns else np.zeros((n_samples, 0))
        return design_matrix, term_info

    def _powers_to_string(
        self,
        powers: Tuple[int, ...],
        var_names: List[str],
    ) -> str:
        """将幂次组合转换为字符串表示。

        Args:
            powers: 幂次元组。
            var_names: 变量名列表。

        Returns:
            字符串表示，如 "x1**2*x2"。
        """
        parts = []
        for var_name, power in zip(var_names, powers):
            if power == 0:
                continue
            elif power == 1:
                parts.append(var_name)
            else:
                parts.append(f"{var_name}**{power}")

        return "*".join(parts) if parts else "1"
