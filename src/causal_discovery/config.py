"""
统一配置类

与YAML配置文件对应的配置类，支持所有工作流
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DiscoveryConfig:
    """因果发现配置"""
    symptom_threshold: float = 0.10
    medicine_threshold: float = 0.10
    diagnosis_threshold: float = 0.10
    alpha: float = 0.05
    independence_test: str = "fisherz"
    depth: int = -1
    apply_constraints: bool = True
    remove_cycles: bool = True


@dataclass
class EffectConfig:
    """因果效应配置"""
    min_correlation: float = 0.0
    min_sample_size: int = 20
    method: str = "logistic_ovr"

    @dataclass
    class BootstrapConfig:
        enable: bool = True
        n_iterations: int = 50
        confidence_level: float = 0.95

    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


@dataclass
class RecommendationConfig:
    """因果推荐配置"""
    threshold_positive: float = 0.05
    threshold_negative: float = -0.05
    top_k: int = 5
    max_paths: int = 5


@dataclass
class NPCConfig:
    """
    Causality-NPC 统一配置类

    与 config/base.yaml 结构完全对应
    """

    # 通用配置
    project_name: str = "Causality-NPC"
    version: str = "2.0"
    raw_data_path: str = "Data/raw/npc_full_with_symptoms.csv"
    outputs_root: str = "outputs"
    experiment_name: str = "default"

    # 各工作流配置
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    effect: EffectConfig = field(default_factory=EffectConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)

    @property
    def discovery_output_dir(self) -> Path:
        """因果发现输出目录"""
        return Path(self.outputs_root) / self.experiment_name / "causal_discovery"

    @property
    def effect_output_dir(self) -> Path:
        """因果效应输出目录"""
        return Path(self.outputs_root) / self.experiment_name / "causal_effects"

    @property
    def recommendation_output_dir(self) -> Path:
        """因果推荐输出目录"""
        return Path(self.outputs_root) / self.experiment_name / "causal_recommendation"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'NPCConfig':
        """
        从YAML文件加载配置

        Parameters
        ----------
        yaml_path : str
            YAML文件路径

        Returns
        -------
        NPCConfig
            配置对象
        """
        import yaml

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 处理配置继承
        if 'extends' in data:
            base_path = Path(yaml_path).parent / data['extends']
            with open(base_path, 'r', encoding='utf-8') as f:
                base_data = yaml.safe_load(f)

            # 合并配置：子配置覆盖父配置
            def deep_merge(base, override):
                """深度合并字典"""
                result = base.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result

            data = deep_merge(base_data, data)

        # 提取实验名称（从文件路径或配置中）
        if 'experiment' in data and 'name' in data['experiment']:
            exp_name = data['experiment']['name']
        else:
            # 从文件路径推断
            path_obj = Path(yaml_path)
            if path_obj.parent.name == 'experiments':
                exp_name = path_obj.stem
            else:
                exp_name = "default"

        return cls(
            project_name=data['project']['name'],
            version=data['project']['version'],
            raw_data_path=data['paths']['raw_data'],
            outputs_root=data['paths']['outputs_root'],
            experiment_name=exp_name,
            discovery=cls._parse_discovery_config(data.get('discovery', {})),
            effect=cls._parse_effect_config(data.get('effect', {})),
            recommendation=cls._parse_recommendation_config(data.get('recommendation', {})),
        )

    @classmethod
    def _parse_discovery_config(cls, data: dict) -> DiscoveryConfig:
        """解析发现配置"""
        return DiscoveryConfig(
            symptom_threshold=data.get('symptom_threshold', 0.10),
            medicine_threshold=data.get('medicine_threshold', 0.10),
            diagnosis_threshold=data.get('diagnosis_threshold', 0.10),
            alpha=data.get('alpha', 0.05),
            independence_test=data.get('independence_test', 'fisherz'),
            depth=data.get('depth', -1),
            apply_constraints=data.get('apply_constraints', True),
            remove_cycles=data.get('remove_cycles', True),
        )

    @classmethod
    def _parse_effect_config(cls, data: dict) -> EffectConfig:
        """解析效应配置"""
        bootstrap_data = data.get('bootstrap', {})
        return EffectConfig(
            min_correlation=data.get('min_correlation', 0.0),
            min_sample_size=data.get('min_sample_size', 20),
            method=data.get('method', 'logistic_ovr'),
            bootstrap=EffectConfig.BootstrapConfig(
                enable=bootstrap_data.get('enable', True),
                n_iterations=bootstrap_data.get('n_iterations', 50),
                confidence_level=bootstrap_data.get('confidence_level', 0.95),
            ),
        )

    @classmethod
    def _parse_recommendation_config(cls, data: dict) -> RecommendationConfig:
        """解析推荐配置"""
        return RecommendationConfig(
            threshold_positive=data.get('threshold_positive', 0.05),
            threshold_negative=data.get('threshold_negative', -0.05),
            top_k=data.get('top_k', 5),
            max_paths=data.get('max_paths', 5),
        )
