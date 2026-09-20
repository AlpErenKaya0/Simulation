'''
KISITLAR VE SORULAR: 
* Q-learning mi kullanacağız sarsa mı ? 
* Dronelar eşit mi olacak yoksa 
bir aralıkta değerlere göre özelliklere sahip dronelar mı yapalım?
 Örneğin dronelar 1 birim ile 4 birim arasında hareket etsin gibi
* tüm dronelar nereye gideceklerini tek bir algoritma tarafından mı öğrenecek? 
* droneların belief stateleri olacak mı? 
* ortamda stokastik bazı engeller olacak mı? 
* hem tek bağlantıda tüm droneların birbirine erişebilmesi hem de birbirlerine erişmesi
mi gerekli yoksa sadece birbirlerine erişebileceklerini göstermek yeterli mi ? 
* bu dronelar hedef noktaları bilerek mi ilerliyor yoksa keşif mi ediyorlar?
* model nerede eğitilecek? 
*simulasyonda kullanılan yapı kaç iterasyon boyunca eğitilmiş olacak? 
*haberleşme maliyeti nasıl hesaplanacak?
*enerji verimliliği nasıl hesaplanacak? 
*reinforcement learning'in RL olarak kalabilmesi için constraintlerimiz neler? 
Sadece bir yapı mı kuracağız yoksa yönlendirebilir miyiz biraz? 
*süre kısıtımız var mı? 
'''

from __future__ import annotations

import math

from simulation_core.algorithm_api import (
    AlgorithmPlan,
    GoalAssignment,
    SimulationAlgorithm,
    SimulationContext,
)


class ReinforcementAlgorithm(SimulationAlgorithm):
    name = "reinforcement"
    display_name = "Reinforcement"
    description = "Basit hedef seçimi tabanli kuvvetlendirme benzeri algoritma."

    def build_plan(self, context: SimulationContext) -> AlgorithmPlan:
        assignments = []
        free_goals = list(context.free_goals())

        for uav in context.free_uavs():
            if not free_goals:
                break

            target = min(
                free_goals,
                key=lambda goal: self.distance_between(uav.position, goal.position),
            )
            assignments.append(GoalAssignment(uav_id=uav.id, goal_id=target.id))
            free_goals.remove(target)

        return AlgorithmPlan(goal_assignments=tuple(assignments))

    @staticmethod
    def distance_between(first, second) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)


ALGORITHMS = [ReinforcementAlgorithm]