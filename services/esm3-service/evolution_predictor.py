"""
Evolution Predictor for EvoHuman.AI ESM3 Service
Predicts evolutionary pathways and optimization strategies
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
import asyncio
from itertools import combinations

from .esm3_engine import ESM3Engine


logger = structlog.get_logger("evolution-predictor")


class EvolutionPredictor:
    """Predict evolutionary pathways for protein optimization"""
    
    def __init__(self, esm3_engine: ESM3Engine, config: Dict[str, Any]):
        self.engine = esm3_engine
        self.config = config
        
        # Amino acid properties for evolution analysis
        self.aa_properties = {
            'A': {'hydrophobic': 0.31, 'volume': 67, 'flexibility': 0.36},
            'R': {'hydrophobic': -1.01, 'volume': 148, 'flexibility': 0.53},
            'N': {'hydrophobic': -0.60, 'volume': 96, 'flexibility': 0.46},
            'D': {'hydrophobic': -0.77, 'volume': 91, 'flexibility': 0.51},
            'C': {'hydrophobic': 1.54, 'volume': 86, 'flexibility': 0.35},
            'Q': {'hydrophobic': -0.22, 'volume': 114, 'flexibility': 0.49},
            'E': {'hydrophobic': -0.64, 'volume': 109, 'flexibility': 0.50},
            'G': {'hydrophobic': 0.0, 'volume': 48, 'flexibility': 0.54},
            'H': {'hydrophobic': 0.13, 'volume': 118, 'flexibility': 0.49},
            'I': {'hydrophobic': 1.80, 'volume': 124, 'flexibility': 0.30},
            'L': {'hydrophobic': 1.70, 'volume': 124, 'flexibility': 0.37},
            'K': {'hydrophobic': -0.99, 'volume': 135, 'flexibility': 0.52},
            'M': {'hydrophobic': 1.23, 'volume': 124, 'flexibility': 0.42},
            'F': {'hydrophobic': 1.79, 'volume': 135, 'flexibility': 0.31},
            'P': {'hydrophobic': 0.72, 'volume': 90, 'flexibility': 0.31},
            'S': {'hydrophobic': -0.04, 'volume': 73, 'flexibility': 0.51},
            'T': {'hydrophobic': 0.26, 'volume': 93, 'flexibility': 0.44},
            'W': {'hydrophobic': 2.25, 'volume': 163, 'flexibility': 0.31},
            'Y': {'hydrophobic': 0.96, 'volume': 141, 'flexibility': 0.42},
            'V': {'hydrophobic': 1.22, 'volume': 105, 'flexibility': 0.33}
        }
    
    async def analyze_pathways(
        self,
        sequence: str,
        target_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze evolutionary pathways for protein optimization"""
        
        logger.info("Starting evolution pathway analysis", 
                   sequence_length=len(sequence),
                   targets=list(target_properties.keys()))
        
        try:
            # Get baseline structure prediction
            baseline_result = await self.engine.predict_structure(sequence, return_contacts=True)
            baseline_confidence = baseline_result["overall_confidence"]
            
            # Identify optimization targets
            pathways = await self._generate_optimization_pathways(
                sequence, 
                baseline_result, 
                target_properties
            )
            
            # Rank pathways by potential
            ranked_pathways = self._rank_pathways(pathways, target_properties)
            
            # Generate specific recommendations
            recommendations = self._generate_recommendations(
                sequence, 
                baseline_result, 
                target_properties
            )
            
            return {
                "evolutionary_pathways": ranked_pathways,
                "baseline_confidence": baseline_confidence,
                "optimization_potential": self._assess_optimization_potential(baseline_result),
                "target_properties": target_properties,
                "recommendations": recommendations,
                "analysis_method": "ESM3-guided evolutionary pathway prediction"
            }
            
        except Exception as e:
            logger.error("Evolution pathway analysis failed", error=str(e))
            raise
    
    async def _generate_optimization_pathways(
        self,
        sequence: str,
        baseline_result: Dict[str, Any],
        target_properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate potential optimization pathways"""
        
        pathways = []
        confidence_scores = baseline_result["confidence_scores"]
        
        # Pathway 1: Stability optimization
        if target_properties.get("stability", False) or target_properties.get("thermal_stability", False):
            stability_pathway = await self._generate_stability_pathway(sequence, confidence_scores)
            pathways.append(stability_pathway)
        
        # Pathway 2: Activity optimization
        if target_properties.get("activity", False) or target_properties.get("catalytic_activity", False):
            activity_pathway = await self._generate_activity_pathway(sequence, confidence_scores)
            pathways.append(activity_pathway)
        
        # Pathway 3: Solubility optimization
        if target_properties.get("solubility", False):
            solubility_pathway = await self._generate_solubility_pathway(sequence)
            pathways.append(solubility_pathway)
        
        # Pathway 4: Binding affinity optimization
        if target_properties.get("binding_affinity", False):
            binding_pathway = await self._generate_binding_pathway(sequence, confidence_scores)
            pathways.append(binding_pathway)
        
        return pathways
    
    async def _generate_stability_pathway(
        self,
        sequence: str,
        confidence_scores: List[float]
    ) -> Dict[str, Any]:
        """Generate stability optimization pathway"""
        
        # Identify low-confidence regions for stabilization
        mean_confidence = np.mean(confidence_scores)
        std_confidence = np.std(confidence_scores)
        threshold = mean_confidence - 0.5 * std_confidence
        
        unstable_positions = [
            i for i, score in enumerate(confidence_scores) 
            if score < threshold and i < len(sequence)
        ]
        
        # Generate stabilizing mutations
        mutations = []
        for pos in unstable_positions[:5]:  # Top 5 positions
            current_aa = sequence[pos]
            # Suggest mutations to more stable amino acids
            stable_candidates = ['A', 'V', 'L', 'I']  # Generally stable amino acids
            for candidate in stable_candidates:
                if candidate != current_aa:
                    mutations.append(f"{current_aa}{pos+1}{candidate}")
                    break
        
        # Estimate improvement potential
        improvement_potential = min(0.8, len(unstable_positions) * 0.05)
        
        return {
            "pathway_id": "stability_optimization",
            "pathway_type": "stability",
            "target_positions": unstable_positions,
            "suggested_mutations": mutations,
            "predicted_improvement": improvement_potential,
            "confidence": 0.75,
            "description": f"Stabilize {len(unstable_positions)} low-confidence regions",
            "estimated_steps": len(mutations),
            "priority": "high" if improvement_potential > 0.3 else "medium"
        }
    
    async def _generate_activity_pathway(
        self,
        sequence: str,
        confidence_scores: List[float]
    ) -> Dict[str, Any]:
        """Generate activity optimization pathway"""
        
        # Focus on flexible regions that might be active sites
        high_flexibility_positions = []
        
        for i, aa in enumerate(sequence):
            if aa in self.aa_properties:
                flexibility = self.aa_properties[aa]['flexibility']
                if flexibility > 0.45 and i < len(confidence_scores):
                    if confidence_scores[i] > 0.6:  # Confident but flexible
                        high_flexibility_positions.append(i)
        
        # Generate activity-enhancing mutations
        mutations = []
        for pos in high_flexibility_positions[:3]:  # Top 3 positions
            current_aa = sequence[pos]
            # Suggest mutations to catalytically active amino acids
            active_candidates = ['H', 'D', 'E', 'R', 'K', 'S', 'T', 'Y']
            for candidate in active_candidates:
                if candidate != current_aa:
                    mutations.append(f"{current_aa}{pos+1}{candidate}")
                    break
        
        improvement_potential = min(0.6, len(high_flexibility_positions) * 0.08)
        
        return {
            "pathway_id": "activity_optimization",
            "pathway_type": "activity",
            "target_positions": high_flexibility_positions,
            "suggested_mutations": mutations,
            "predicted_improvement": improvement_potential,
            "confidence": 0.65,
            "description": f"Enhance activity through {len(mutations)} strategic mutations",
            "estimated_steps": len(mutations),
            "priority": "medium"
        }
    
    async def _generate_solubility_pathway(self, sequence: str) -> Dict[str, Any]:
        """Generate solubility optimization pathway"""
        
        # Identify hydrophobic surface residues
        hydrophobic_positions = []
        
        for i, aa in enumerate(sequence):
            if aa in self.aa_properties:
                hydrophobicity = self.aa_properties[aa]['hydrophobic']
                if hydrophobicity > 1.0:  # Highly hydrophobic
                    hydrophobic_positions.append(i)
        
        # Generate solubility-enhancing mutations
        mutations = []
        for pos in hydrophobic_positions[:4]:  # Top 4 positions
            current_aa = sequence[pos]
            # Suggest mutations to hydrophilic amino acids
            hydrophilic_candidates = ['D', 'E', 'K', 'R', 'N', 'Q', 'S', 'T']
            for candidate in hydrophilic_candidates:
                if candidate != current_aa:
                    mutations.append(f"{current_aa}{pos+1}{candidate}")
                    break
        
        improvement_potential = min(0.7, len(hydrophobic_positions) * 0.06)
        
        return {
            "pathway_id": "solubility_optimization",
            "pathway_type": "solubility",
            "target_positions": hydrophobic_positions,
            "suggested_mutations": mutations,
            "predicted_improvement": improvement_potential,
            "confidence": 0.70,
            "description": f"Improve solubility by modifying {len(mutations)} hydrophobic residues",
            "estimated_steps": len(mutations),
            "priority": "medium"
        }
    
    async def _generate_binding_pathway(
        self,
        sequence: str,
        confidence_scores: List[float]
    ) -> Dict[str, Any]:
        """Generate binding affinity optimization pathway"""
        
        # Identify potential binding regions (moderate confidence, surface-like)
        binding_positions = []
        
        for i, score in enumerate(confidence_scores):
            if 0.5 < score < 0.8 and i < len(sequence):  # Moderate confidence
                binding_positions.append(i)
        
        # Generate binding-enhancing mutations
        mutations = []
        for pos in binding_positions[:3]:  # Top 3 positions
            current_aa = sequence[pos]
            # Suggest mutations to binding-favorable amino acids
            binding_candidates = ['R', 'K', 'H', 'Y', 'W', 'F']
            for candidate in binding_candidates:
                if candidate != current_aa:
                    mutations.append(f"{current_aa}{pos+1}{candidate}")
                    break
        
        improvement_potential = min(0.5, len(binding_positions) * 0.07)
        
        return {
            "pathway_id": "binding_optimization",
            "pathway_type": "binding_affinity",
            "target_positions": binding_positions,
            "suggested_mutations": mutations,
            "predicted_improvement": improvement_potential,
            "confidence": 0.60,
            "description": f"Enhance binding through {len(mutations)} interface mutations",
            "estimated_steps": len(mutations),
            "priority": "low"
        }
    
    def _rank_pathways(
        self,
        pathways: List[Dict[str, Any]],
        target_properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank pathways by potential and relevance"""
        
        # Calculate ranking scores
        for pathway in pathways:
            score = 0.0
            
            # Base score from predicted improvement
            score += pathway["predicted_improvement"] * 100
            
            # Bonus for high confidence
            score += pathway["confidence"] * 20
            
            # Bonus for matching target properties
            pathway_type = pathway["pathway_type"]
            if pathway_type in target_properties:
                score += 30
            
            # Penalty for high complexity
            score -= pathway["estimated_steps"] * 2
            
            pathway["ranking_score"] = score
        
        # Sort by ranking score
        return sorted(pathways, key=lambda x: x["ranking_score"], reverse=True)
    
    def _assess_optimization_potential(self, baseline_result: Dict[str, Any]) -> str:
        """Assess overall optimization potential"""
        confidence = baseline_result["overall_confidence"]
        confidence_scores = baseline_result["confidence_scores"]
        
        low_confidence_fraction = sum(1 for score in confidence_scores if score < 0.7) / len(confidence_scores)
        
        if confidence < 0.6 or low_confidence_fraction > 0.3:
            return "high"
        elif confidence < 0.8 or low_confidence_fraction > 0.15:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(
        self,
        sequence: str,
        baseline_result: Dict[str, Any],
        target_properties: Dict[str, Any]
    ) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        confidence = baseline_result["overall_confidence"]
        confidence_scores = baseline_result["confidence_scores"]
        
        # General recommendations based on confidence
        if confidence < 0.7:
            recommendations.append("Focus on overall structural stabilization before specific optimizations")
        
        # Specific recommendations based on targets
        if target_properties.get("stability", False):
            low_conf_count = sum(1 for score in confidence_scores if score < 0.6)
            if low_conf_count > 0:
                recommendations.append(f"Prioritize stabilizing {low_conf_count} low-confidence regions")
        
        if target_properties.get("activity", False):
            recommendations.append("Consider mutations in flexible loop regions for activity enhancement")
        
        if target_properties.get("solubility", False):
            hydrophobic_count = sum(1 for aa in sequence if aa in ['F', 'W', 'Y', 'L', 'I', 'V'])
            if hydrophobic_count > len(sequence) * 0.4:
                recommendations.append("Reduce surface hydrophobicity to improve solubility")
        
        # Add general best practices
        recommendations.extend([
            "Test mutations individually before combining",
            "Validate predictions with experimental data",
            "Consider evolutionary conservation when selecting mutations"
        ])
        
        return recommendations
