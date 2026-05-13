# 德州扑克AI辅助建议器

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AIAvisor')

from ..game.poker_game import PokerGame, Player, Card
from ..model.ai_agent import PokerAIAgent
from ..model.hand_evaluator import HandEvaluator

class AIAvisor:
    """德州扑克AI辅助建议器，基于训练好的模型提供游戏建议"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = 'dqn',
                 confidence_threshold: float = 0.7):
        """
        初始化AI建议器
        
        Args:
            model_path: 训练好的模型路径
            model_type: 模型类型（'dqn'或'policy_gradient'）
            confidence_threshold: 建议置信度阈值
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # 创建AI代理用于提供建议
        self.agent = PokerAIAgent(agent_type=model_type)
        
        # 如果提供了模型路径，加载模型
        if model_path:
            self.load_model(model_path)
        
        # 手牌评估器
        self.hand_evaluator = HandEvaluator()
    
    def load_model(self, model_path: str) -> bool:
        """加载训练好的模型"""
        try:
            self.agent.load_model(model_path)
            logger.info(f"成功加载模型: {model_path}")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def _calculate_hand_strength(self, hand: List[Card], community_cards: List[Card]) -> Tuple[float, str]:
        """
        计算手牌强度
        
        Args:
            hand: 玩家手牌
            community_cards: 公共牌
            
        Returns:
            手牌强度（0-1之间的浮点数）和手牌类型
        """
        # 如果有公共牌，使用它们进行评估
        all_cards = hand + community_cards
        
        # 计算当前手牌强度
        hand_strength, hand_type = self.hand_evaluator.evaluate_hand(all_cards)
        
        # 如果公共牌少于3张，基于可能的组合进行模拟评估
        if len(community_cards) < 3:
            # 对于翻牌前，使用预定义的手牌强度
            if len(community_cards) == 0:
                return self._preflop_hand_strength(hand), hand_type
            
            # 模拟后续可能的公共牌组合，计算平均强度
            num_sims = 1000
            sim_results = []
            
            # 获取牌堆中剩余的牌
            remaining_cards = self.hand_evaluator.get_remaining_cards(all_cards)
            
            # 计算需要模拟的公共牌数量
            needed = 5 - len(community_cards)
            
            for _ in range(min(num_sims, 1000)):
                # 模拟剩余公共牌
                if len(remaining_cards) >= needed:
                    sim_community = random.sample(remaining_cards, needed)
                    # 组合所有牌进行评估
                    sim_all_cards = hand + community_cards + sim_community
                    strength, _ = self.hand_evaluator.evaluate_hand(sim_all_cards)
                    sim_results.append(strength)
            
            # 返回平均强度
            if sim_results:
                return np.mean(sim_results), hand_type
        
        return hand_strength, hand_type
    
    def _preflop_hand_strength(self, hand: List[Card]) -> float:
        """计算翻牌前的手牌强度"""
        # 简化的翻牌前手牌强度评估
        # 对子、同花连牌等会有更高的基础强度
        
        ranks = sorted([card.rank for card in hand])
        suits = [card.suit for card in hand]
        
        # 对子
        if ranks[0] == ranks[1]:
            # 对子强度：A>A>K>Q>J>10>9>8>7>6>5>4>3>2
            rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                          '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            pair_strength = rank_values[ranks[0]] / 14
            return 0.6 + pair_strength * 0.4
        
        # 同花
        if suits[0] == suits[1]:
            suited_bonus = 0.1
        else:
            suited_bonus = 0.0
        
        # 连牌
        is_consecutive = abs(self.hand_evaluator.get_rank_value(ranks[0]) - 
                           self.hand_evaluator.get_rank_value(ranks[1])) == 1
        consecutive_bonus = 0.15 if is_consecutive else 0.0
        
        # 高牌强度
        high_card_strength = (self.hand_evaluator.get_rank_value(max(ranks, key=lambda r: self.hand_evaluator.get_rank_value(r))) / 14) * 0.5
        
        # 低牌强度（对于小连牌）
        low_card_strength = (self.hand_evaluator.get_rank_value(min(ranks, key=lambda r: self.hand_evaluator.get_rank_value(r))) / 14) * 0.2
        
        # 综合强度
        total_strength = 0.3 + high_card_strength - low_card_strength + suited_bonus + consecutive_bonus
        
        return min(max(total_strength, 0.0), 1.0)
    
    def _predict_opponent_actions(self, game: PokerGame, player: Player) -> Dict[str, float]:
        """
        预测对手可能的动作
        
        Args:
            game: 当前游戏状态
            player: 目标对手
            
        Returns:
            对手可能动作的概率分布
        """
        # 简化的对手动作预测
        # 实际应用中可以使用更复杂的对手建模
        
        # 计算对手可能的动作概率
        actions = {'fold': 0.0, 'call': 0.0, 'raise': 0.0}
        
        # 根据对手的筹码量、当前下注、历史行为等因素预测
        opponent_stack = player.stack
        current_bet = game.current_bet
        
        # 基于游戏阶段调整预测
        if game.round_num == 0:  # 翻牌前
            actions['fold'] = 0.5
            actions['call'] = 0.3
            actions['raise'] = 0.2
        elif game.round_num == 1:  # 翻牌
            actions['fold'] = 0.3
            actions['call'] = 0.4
            actions['raise'] = 0.3
        elif game.round_num == 2:  # 转牌
            actions['fold'] = 0.25
            actions['call'] = 0.45
            actions['raise'] = 0.3
        else:  # 河牌
            actions['fold'] = 0.2
            actions['call'] = 0.5
            actions['raise'] = 0.3
        
        # 归一化概率
        total = sum(actions.values())
        for key in actions:
            actions[key] = actions[key] / total if total > 0 else 0.0
        
        return actions
    
    def get_basic_advice(self, game: PokerGame, player: Player) -> Dict[str, Union[str, float, Dict]]:
        """
        获取基础游戏建议
        
        Args:
            game: 当前游戏状态
            player: 当前玩家
            
        Returns:
            包含建议的字典
        """
        # 获取玩家手牌和公共牌
        hand = player.cards
        community_cards = game.community_cards
        
        # 计算手牌强度
        hand_strength, hand_type = self._calculate_hand_strength(hand, community_cards)
        
        # 让AI代理选择动作
        ai_action, ai_amount = self.agent.select_action(game, player)
        
        # 计算预期价值
        expected_value = self._calculate_expected_value(game, player, ai_action, ai_amount)
        
        # 预测对手动作
        opponent_predictions = {}
        for p in game.players:
            if p != player and p.is_active:
                opponent_predictions[p.name] = self._predict_opponent_actions(game, p)
        
        # 构建建议
        advice = {
            'hand_strength': hand_strength,
            'hand_type': hand_type,
            'recommended_action': ai_action,
            'recommended_amount': ai_amount,
            'expected_value': expected_value,
            'confidence': self._calculate_confidence(game, player, hand_strength, expected_value),
            'opponent_predictions': opponent_predictions,
            'game_stage': self._get_game_stage_name(game.round_num),
            'pot_odds': self._calculate_pot_odds(game, player)
        }
        
        return advice
    
    def _calculate_expected_value(self, game: PokerGame, player: Player, 
                                 action: str, amount: float) -> float:
        """
        计算预期价值
        """
        # 简化的预期价值计算
        pot_size = game.pot
        
        if action == 'fold':
            return -player.bet
        elif action == 'call':
            # 基于手牌强度估计获胜概率
            hand_strength, _ = self._calculate_hand_strength(player.cards, game.community_cards)
            win_probability = min(hand_strength * 1.2, 1.0)  # 对手牌强度进行一些调整
            return win_probability * pot_size - amount
        elif action == 'raise':
            # 对于加注，考虑对手的弃牌率和我们的获胜概率
            hand_strength, _ = self._calculate_hand_strength(player.cards, game.community_cards)
            win_probability = min(hand_strength * 1.5, 1.0)
            
            # 估计对手弃牌率（基于加注大小）
            fold_probability = min(amount / pot_size * 0.5, 0.7)
            
            # 综合计算预期价值
            ev_if_fold = pot_size
            ev_if_call = win_probability * (pot_size + amount) - amount
            
            return fold_probability * ev_if_fold + (1 - fold_probability) * ev_if_call
        
        return 0.0
    
    def _calculate_confidence(self, game: PokerGame, player: Player, 
                             hand_strength: float, expected_value: float) -> float:
        """
        计算建议的置信度
        """
        # 基于手牌强度、预期价值、游戏阶段等因素计算置信度
        base_confidence = min(hand_strength * 0.8 + 0.2, 1.0)
        
        # 基于预期价值调整
        if expected_value > 0:
            confidence_bonus = min(expected_value / game.pot * 0.5, 0.3)
        else:
            confidence_bonus = max(expected_value / game.pot * 0.5, -0.3)
        
        # 基于信息完整性调整（公共牌越多，信息越完整）
        info_factor = min(len(game.community_cards) / 5 * 0.2 + 0.8, 1.0)
        
        confidence = (base_confidence + confidence_bonus) * info_factor
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_pot_odds(self, game: PokerGame, player: Player) -> float:
        """
        计算底池赔率
        """
        amount_to_call = game.current_bet - player.bet
        if amount_to_call <= 0:
            return 0.0
        
        return game.pot / amount_to_call
    
    def _get_game_stage_name(self, round_num: int) -> str:
        """
        获取游戏阶段名称
        """
        stages = ['翻牌前', '翻牌', '转牌', '河牌']
        if 0 <= round_num < len(stages):
            return stages[round_num]
        return '未知阶段'
    
    def get_detailed_advice(self, game: PokerGame, player: Player) -> Dict[str, Union[str, float, Dict, List]]:
        """
        获取详细的游戏建议，包括分析和可视化数据
        
        Args:
            game: 当前游戏状态
            player: 当前玩家
            
        Returns:
            包含详细建议的字典
        """
        # 获取基础建议
        basic_advice = self.get_basic_advice(game, player)
        
        # 添加更详细的分析
        detailed_advice = basic_advice.copy()
        
        # 分析各种可能动作的结果
        action_analysis = self._analyze_all_actions(game, player)
        detailed_advice['action_analysis'] = action_analysis
        
        # 添加胜率分析
        win_rate_analysis = self._analyze_win_rates(game, player)
        detailed_advice['win_rate_analysis'] = win_rate_analysis
        
        # 添加决策理由
        detailed_advice['reasoning'] = self._generate_reasoning(game, player, basic_advice)
        
        # 添加风险评估
        detailed_advice['risk_assessment'] = self._assess_risk(game, player, basic_advice)
        
        # 生成可视化数据（用于后续可能的图表展示）
        detailed_advice['visualization_data'] = self._generate_visualization_data(game, player, detailed_advice)
        
        return detailed_advice
    
    def _analyze_all_actions(self, game: PokerGame, player: Player) -> Dict[str, Dict[str, float]]:
        """
        分析所有可能动作的预期结果
        """
        actions_to_analyze = ['fold', 'call']
        
        # 为加注生成几个可能的金额选项
        current_bet = game.current_bet
        min_raise = current_bet * 2  # 假设最小加注是当前下注的两倍
        max_raise = player.stack  # 最大加注是全下
        
        # 添加几个不同级别的加注选项
        raise_amounts = [min_raise, min_raise * 2, (min_raise + max_raise) // 2, max_raise]
        raise_amounts = [amount for amount in raise_amounts if amount > current_bet]
        
        analysis = {}
        
        # 分析弃牌
        analysis['fold'] = {
            'expected_value': self._calculate_expected_value(game, player, 'fold', 0),
            'risk_level': 0.1,  # 低风险
            'win_probability': 0.0
        }
        
        # 分析跟注
        call_amount = max(0, current_bet - player.bet)
        hand_strength, _ = self._calculate_hand_strength(player.cards, game.community_cards)
        analysis['call'] = {
            'expected_value': self._calculate_expected_value(game, player, 'call', call_amount),
            'risk_level': min(hand_strength * 0.5 + 0.3, 1.0),
            'win_probability': min(hand_strength * 1.2, 1.0),
            'amount': call_amount
        }
        
        # 分析加注
        for raise_amount in raise_amounts:
            action_key = f'raise_{raise_amount}'
            hand_strength, _ = self._calculate_hand_strength(player.cards, game.community_cards)
            analysis[action_key] = {
                'expected_value': self._calculate_expected_value(game, player, 'raise', raise_amount),
                'risk_level': min((1 - hand_strength) * 0.5 + 0.5, 1.0),
                'win_probability': min(hand_strength * 1.5, 1.0),
                'amount': raise_amount,
                'fold_probability': min(raise_amount / game.pot * 0.5, 0.7)
            }
        
        return analysis
    
    def _analyze_win_rates(self, game: PokerGame, player: Player) -> Dict[str, float]:
        """
        分析不同场景下的胜率
        """
        # 基于手牌强度和游戏阶段的胜率分析
        hand_strength, _ = self._calculate_hand_strength(player.cards, game.community_cards)
        
        # 基本胜率
        base_win_rate = hand_strength * 100
        
        # 根据游戏阶段调整
        stage_multipliers = [1.0, 0.8, 0.7, 0.6]  # 翻牌前、翻牌、转牌、河牌
        stage_multiplier = stage_multipliers[min(game.round_num, 3)]
        
        # 根据对手数量调整
        num_active_opponents = sum(1 for p in game.players if p != player and p.is_active)
        opponent_multiplier = 1.0 / (num_active_opponents * 0.3 + 1.0)
        
        # 计算各种情况下的胜率估计
        win_rates = {
            'current_estimated_win_rate': base_win_rate * stage_multiplier * opponent_multiplier,
            'against_single_opponent': base_win_rate * stage_multiplier,
            'best_case_win_rate': min(base_win_rate * 1.2, 95.0),
            'worst_case_win_rate': max(base_win_rate * 0.5, 5.0),
            'with_remaining_cards': self._estimate_win_rate_with_remaining_cards(game, player)
        }
        
        return win_rates
    
    def _estimate_win_rate_with_remaining_cards(self, game: PokerGame, player: Player) -> float:
        """
        估计考虑剩余未发牌的胜率
        """
        # 简化处理，返回一个基于当前手牌强度和游戏阶段的估计值
        hand_strength, _ = self._calculate_hand_strength(player.cards, game.community_cards)
        remaining_cards = 5 - len(game.community_cards)
        
        # 剩余牌越多，不确定性越大
        uncertainty_factor = 1.0 - (remaining_cards / 5) * 0.4
        
        return hand_strength * uncertainty_factor * 100
    
    def _generate_reasoning(self, game: PokerGame, player: Player, 
                          advice: Dict[str, Union[str, float, Dict]]) -> str:
        """
        生成决策建议的理由说明
        """
        action = advice['recommended_action']
        hand_strength = advice['hand_strength']
        expected_value = advice['expected_value']
        confidence = advice['confidence']
        game_stage = advice['game_stage']
        
        # 基于这些因素生成自然语言的理由说明
        reasoning = f"基于当前{game_stage}阶段的分析，"
        
        if hand_strength > 0.7:
            reasoning += "您的手牌强度较高（%.1f%%），" % (hand_strength * 100)
        elif hand_strength > 0.4:
            reasoning += "您的手牌强度中等（%.1f%%），" % (hand_strength * 100)
        else:
            reasoning += "您的手牌强度较低（%.1f%%），" % (hand_strength * 100)
        
        if expected_value > 0:
            reasoning += "预期价值为正（+%.2f），" % expected_value
        else:
            reasoning += "预期价值为负（%.2f），" % expected_value
        
        if action == 'fold':
            reasoning += "建议弃牌以避免不必要的损失。"
        elif action == 'call':
            reasoning += f"建议跟注（%.2f）以继续游戏。" % advice['recommended_amount']
        elif action == 'raise':
            reasoning += f"建议加注到%.2f以建立更大的底池或迫使对手弃牌。" % advice['recommended_amount']
        
        if confidence > 0.8:
            reasoning += "此建议的置信度很高（%.1f%%）。" % (confidence * 100)
        elif confidence > 0.5:
            reasoning += "此建议的置信度中等（%.1f%%）。" % (confidence * 100)
        else:
            reasoning += "此建议的置信度较低（%.1f%%），请谨慎决策。" % (confidence * 100)
        
        return reasoning
    
    def _assess_risk(self, game: PokerGame, player: Player, 
                   advice: Dict[str, Union[str, float, Dict]]) -> Dict[str, float]:
        """
        评估决策的风险
        """
        action = advice['recommended_action']
        hand_strength = advice['hand_strength']
        
        # 计算风险相关指标
        risk_level = 0.0
        potential_loss = 0.0
        reward_risk_ratio = 1.0
        
        if action == 'fold':
            risk_level = 0.1  # 低风险
            potential_loss = 0.0
        elif action == 'call':
            amount = max(0, game.current_bet - player.bet)
            risk_level = min(1.0 - hand_strength, 0.9)  # 手牌越弱，风险越高
            potential_loss = amount
        elif action == 'raise':
            amount = advice['recommended_amount']
            risk_level = min(1.0 - (hand_strength * 0.5), 0.95)  # 加注风险更高
            potential_loss = amount
        
        # 计算风险回报比
        if risk_level > 0:
            reward_risk_ratio = advice['expected_value'] / risk_level if advice['expected_value'] > 0 else -1
        
        # 评估破产风险
        bankruptcy_risk = 0.0
        total_investment = player.bet + (advice['recommended_amount'] if action == 'raise' else 0)
        if total_investment > player.stack * 0.75:
            bankruptcy_risk = (total_investment / player.stack) * 0.5
        
        risk_assessment = {
            'risk_level': risk_level,  # 0-1，越高风险越大
            'potential_loss': potential_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'bankruptcy_risk': bankruptcy_risk,
            'is_risky_decision': risk_level > 0.7
        }
        
        return risk_assessment
    
    def _generate_visualization_data(self, game: PokerGame, player: Player, 
                                   advice: Dict[str, Union[str, float, Dict, List]]) -> Dict:
        """
        生成用于可视化的数据
        """
        # 提取手牌强度分布数据
        hand_strength_data = []
        for r in range(10):
            strength = r / 10.0
            # 模拟不同手牌强度的频率分布
            # 这里简化处理，实际可以基于历史数据或更复杂的模型
            if strength < 0.2:
                freq = 0.4
            elif strength < 0.4:
                freq = 0.3
            elif strength < 0.6:
                freq = 0.15
            elif strength < 0.8:
                freq = 0.1
            else:
                freq = 0.05
            hand_strength_data.append((strength, freq))
        
        # 提取动作分析数据
        action_data = []
        for action, data in advice['action_analysis'].items():
            if action in ['fold', 'call'] or action.startswith('raise_'):
                action_data.append({
                    'action': action,
                    'expected_value': data['expected_value'],
                    'risk_level': data['risk_level'] if 'risk_level' in data else 0.0
                })
        
        # 提取胜率分析数据
        win_rate_data = {
            'current': advice['win_rate_analysis']['current_estimated_win_rate'],
            'single_opponent': advice['win_rate_analysis']['against_single_opponent'],
            'best_case': advice['win_rate_analysis']['best_case_win_rate'],
            'worst_case': advice['win_rate_analysis']['worst_case_win_rate']
        }
        
        visualization_data = {
            'hand_strength_distribution': hand_strength_data,
            'action_analysis': action_data,
            'win_rate_comparison': win_rate_data,
            'pot_evolution': [game.pot],  # 这里可以添加历史底池大小
            'stack_evolution': {player.name: player.stack}  # 玩家筹码变化
        }
        
        return visualization_data
    
    def save_advice_to_file(self, advice: Dict, filename: Optional[str] = None) -> str:
        """
        保存建议到文件
        
        Args:
            advice: 要保存的建议
            filename: 文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        import json
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advice_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs('advice_logs', exist_ok=True)
        filepath = os.path.join('advice_logs', filename)
        
        try:
            # 将numpy类型转换为Python原生类型以便序列化
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            # 转换数据
            serializable_advice = convert_numpy(advice)
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_advice, f, ensure_ascii=False, indent=2)
            
            logger.info(f"建议已保存到: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存建议失败: {e}")
            return ""

# 导入random模块（在_preflop_hand_strength方法中使用）
import random


# ----- 向后兼容别名 -----
# 历史代码中类名拼写为 AIAvisor（缺少 'd'），新代码请使用 AIAdvisor，
# 但为不破坏既有引用同时保留两个名字。
AIAdvisor = AIAvisor

__all__ = ["AIAvisor", "AIAdvisor"]

# 示例使用
if __name__ == "__main__":
    # 这里需要创建游戏实例和玩家才能使用建议器
    # 为了演示，这里仅展示如何初始化和加载模型
    print("德州扑克AI建议器示例")
    print("请在实际游戏环境中使用此模块")