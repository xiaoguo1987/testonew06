# 检查点保存和加载工具

import os
import json
import pickle
import logging
import tensorflow as tf
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CheckpointManager')

def save_checkpoint(model, filepath: str, metadata: dict = None) -> bool:
    """
    保存模型检查点
    
    Args:
        model: 要保存的模型对象
        filepath: 保存路径
        metadata: 附加的元数据信息
    
    Returns:
        是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型权重
        if hasattr(model, 'save_weights'):
            # TensorFlow模型
            model.save_weights(filepath + '.weights')
        elif hasattr(model, 'save'):
            # 使用模型的save方法
            model.save(filepath)
        else:
            # 使用pickle保存整个模型
            with open(filepath + '.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # 保存元数据
        if metadata:
            metadata_file = filepath + '.metadata.json'
            metadata['timestamp'] = datetime.now().isoformat()
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功保存检查点到: {filepath}")
        return True
    except Exception as e:
        logger.error(f"保存检查点失败: {e}")
        return False

def load_checkpoint(model, filepath: str) -> tuple:
    """
    加载模型检查点
    
    Args:
        model: 要加载权重的模型对象
        filepath: 加载路径
    
    Returns:
        (是否加载成功, 元数据)
    """
    try:
        # 尝试不同的加载方式
        if os.path.exists(filepath + '.weights'):
            # 加载TensorFlow权重
            if hasattr(model, 'load_weights'):
                model.load_weights(filepath + '.weights')
                logger.info(f"成功加载权重文件: {filepath + '.weights'}")
        elif os.path.exists(filepath + '.pkl'):
            # 加载pickle模型
            with open(filepath + '.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
                # 将加载的模型属性复制到当前模型
                for key, value in loaded_model.__dict__.items():
                    if not key.startswith('_') or key in ['_model', '_target_model']:
                        setattr(model, key, value)
                logger.info(f"成功加载pickle模型: {filepath + '.pkl'}")
        elif os.path.exists(filepath):
            # 尝试使用模型的load方法
            if hasattr(model, 'load'):
                model.load(filepath)
                logger.info(f"成功加载模型: {filepath}")
        else:
            logger.error(f"未找到检查点文件: {filepath}")
            return False, {}
        
        # 加载元数据
        metadata = {}
        metadata_file = filepath + '.metadata.json'
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return True, metadata
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return False, {}

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    查找最新的检查点
    
    Args:
        checkpoint_dir: 检查点目录
    
    Returns:
        最新检查点的路径，如果没有则返回空字符串
    """
    try:
        if not os.path.exists(checkpoint_dir):
            return ""
        
        # 获取所有检查点文件
        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.metadata.json'):
                # 读取时间戳
                with open(os.path.join(checkpoint_dir, filename), 'r') as f:
                    metadata = json.load(f)
                    timestamp = metadata.get('timestamp', '')
                    checkpoint_path = os.path.join(checkpoint_dir, filename[:-len('.metadata.json')])
                    checkpoints.append((timestamp, checkpoint_path))
        
        if not checkpoints:
            return ""
        
        # 按时间戳排序，返回最新的
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]
    except Exception as e:
        logger.error(f"查找最新检查点失败: {e}")
        return ""