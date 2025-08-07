"""
Protein Analyzer for EvoHuman.AI ESM3 Service
High-level protein analysis using ESM3 engine
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
import asyncio
from datetime import datetime

from .esm3_engine import ESM3Engine


logger = structlog.get_logger("protein-analyzer")


class ProteinAnalyzer:
    """High-level protein analysis using ESM3"""
    
    def __init__(self, esm3_engine: ESM3Engine, config: Dict[str, Any]):
        self.engine = esm3_engine
        self.config = config
        self.analysis_cache = {}  # Simple in-memory cache
        
    async def analyze_sequence(
        self,
        sequence: str,
        analysis_type: str = "structure_prediction",
        include_mutations: bool = False,
        include_evolution: bool = False,
        analysis_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive protein sequence analysis"""
        
        logger.info("Starting protein analysis", 
                   analysis_id=analysis_id,
                   sequence_length=len(sequence),
                   analysis_type=analysis_type)
        
        start_time = datetime.utcnow()
        
        try:
            # Base structure prediction
            structure_result = await self.engine.predict_structure(
                sequence=sequence,
                return_contacts=True
            )
            
            # Build result
            result = {
                "sequence_id": analysis_id,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "predicted_structure": self._format_structure_prediction(structure_result),
                "confidence_score": structure_result["overall_confidence"],
                "analysis_type": analysis_type,
                "timestamp": start_time.isoformat()
            }
            
            # Add contact map if available
            if structure_result.get("contacts"):
                result["contact_map"] = {
                    "contacts": structure_result["contacts"],
                    "description": "Predicted residue-residue contacts"
                }
            
            # Add mutation analysis if requested
            if include_mutations:
                mutation_result = await self._analyze_mutation_hotspots(sequence)
                result["mutation_analysis"] = mutation_result
            
            # Add evolution analysis if requested
            if include_evolution:
                evolution_result = await self._analyze_evolutionary_potential(sequence, structure_result)
                result["evolution_suggestion"] = evolution_result
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            result["processing_time"] = processing_time
            result["status"] = "completed"
            
            logger.info("Protein analysis completed", 
                       analysis_id=analysis_id,
                       processing_time=processing_time,
                       confidence=result["confidence_score"])
            
            return result
            
        except Exception as e:
            logger.error("Protein analysis failed", 
                        analysis_id=analysis_id,
                        error=str(e))
            raise
    
    def _format_structure_prediction(self, structure_result: Dict[str, Any]) -> str:
        """Format structure prediction results"""
        confidence = structure_result["overall_confidence"]
        length = structure_result["sequence_length"]
        
        if confidence > 0.9:
            quality = "High confidence"
        elif confidence > 0.7:
            quality = "Medium confidence"
        else:
            quality = "Low confidence"
        
        return f"{quality} structure prediction for {length} residues (confidence: {confidence:.3f})"
    
    async def _analyze_mutation_hotspots(self, sequence: str) -> Dict[str, Any]:
        """Identify mutation hotspots in the protein"""
        try:
            # Generate per-residue confidence scores
            structure_result = await self.engine.predict_structure(sequence, return_contacts=False)
            confidence_scores = structure_result["confidence_scores"]
            
            # Identify low-confidence regions as potential mutation hotspots
            threshold = np.mean(confidence_scores) - np.std(confidence_scores)
            hotspots = [i for i, score in enumerate(confidence_scores) if score < threshold]
            
            # Analyze a few sample mutations at hotspots
            sample_mutations = []
            amino_acids = ['A', 'V', 'L', 'I', 'F', 'Y', 'W', 'S', 'T', 'N', 'Q', 'R', 'K', 'H', 'D', 'E', 'C', 'M', 'P', 'G']
            
            for pos in hotspots[:5]:  # Analyze first 5 hotspots
                if pos < len(sequence):
                    original_aa = sequence[pos]
                    # Try a few different amino acids
                    for new_aa in ['A', 'V', 'L']:  # Conservative mutations
                        if new_aa != original_aa:
                            sample_mutations.append((pos, original_aa, new_aa))
            
            # Analyze sample mutations
            mutation_effects = []
            if sample_mutations:
                mutation_result = await self.engine.analyze_mutations(sequence, sample_mutations[:3])
                mutation_effects = mutation_result.get("mutation_effects", [])
            
            return {
                "hotspots": hotspots,
                "hotspot_count": len(hotspots),
                "stability_effects": f"Identified {len(hotspots)} potential mutation sites",
                "sample_mutations": mutation_effects,
                "analysis_method": "ESM3 confidence-based hotspot detection"
            }
            
        except Exception as e:
            logger.error("Mutation hotspot analysis failed", error=str(e))
            return {
                "hotspots": [],
                "error": str(e)
            }
    
    async def _analyze_evolutionary_potential(self, sequence: str, structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evolutionary optimization potential"""
        try:
            confidence = structure_result["overall_confidence"]
            confidence_scores = structure_result["confidence_scores"]
            
            # Identify regions with optimization potential
            low_confidence_regions = []
            high_confidence_regions = []
            
            for i, score in enumerate(confidence_scores):
                if score < 0.7:
                    low_confidence_regions.append(i)
                elif score > 0.9:
                    high_confidence_regions.append(i)
            
            # Generate evolution suggestions
            pathways = []
            
            if low_confidence_regions:
                pathways.append({
                    "pathway_type": "stability_optimization",
                    "target_regions": low_confidence_regions[:10],  # Top 10 regions
                    "description": "Optimize low-confidence regions for improved stability",
                    "priority": "high"
                })
            
            if len(sequence) > 100:
                pathways.append({
                    "pathway_type": "domain_optimization",
                    "target_regions": list(range(50, min(150, len(sequence)))),
                    "description": "Optimize central domain for enhanced function",
                    "priority": "medium"
                })
            
            # Generate specific optimization targets
            optimization_targets = []
            
            if confidence < 0.8:
                optimization_targets.append("structural_stability")
            
            if len(low_confidence_regions) > len(sequence) * 0.2:
                optimization_targets.append("folding_efficiency")
            
            optimization_targets.extend(["catalytic_activity", "binding_affinity"])
            
            return {
                "pathways": pathways,
                "optimization_targets": optimization_targets,
                "evolutionary_potential": "high" if len(low_confidence_regions) > 10 else "medium",
                "confidence_distribution": {
                    "low_confidence_regions": len(low_confidence_regions),
                    "high_confidence_regions": len(high_confidence_regions),
                    "overall_confidence": confidence
                },
                "recommendations": [
                    "Focus on stabilizing low-confidence regions",
                    "Consider conservative mutations in flexible loops",
                    "Maintain high-confidence structural elements"
                ]
            }
            
        except Exception as e:
            logger.error("Evolution analysis failed", error=str(e))
            return {
                "pathways": [],
                "error": str(e)
            }
    
    async def predict_mutation_effects(
        self,
        sequence: str,
        mutations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict effects of specific mutations"""
        try:
            # Convert mutation format
            mutation_tuples = []
            for mut in mutations:
                if "position" in mut and "from_aa" in mut and "to_aa" in mut:
                    mutation_tuples.append((
                        mut["position"],
                        mut["from_aa"],
                        mut["to_aa"]
                    ))
            
            if not mutation_tuples:
                raise ValueError("No valid mutations provided")
            
            # Analyze mutations using ESM3
            result = await self.engine.analyze_mutations(sequence, mutation_tuples)
            
            # Format results
            formatted_effects = []
            for effect in result["mutation_effects"]:
                if "error" not in effect:
                    formatted_effects.append({
                        "mutation": f"{effect['from_aa']}{effect['position']}{effect['to_aa']}",
                        "stability_change": effect["stability_change"],
                        "confidence": min(0.95, max(0.5, 0.8 + effect["stability_change"] * 0.1)),
                        "effect_category": self._categorize_mutation_effect(effect["stability_change"]),
                        "recommendation": self._get_mutation_recommendation(effect["stability_change"])
                    })
                else:
                    formatted_effects.append({
                        "mutation": f"Position {effect['position']}",
                        "error": effect["error"]
                    })
            
            return {
                "mutation_effects": formatted_effects,
                "wild_type_confidence": result["wild_type_confidence"],
                "total_mutations_analyzed": len(mutation_tuples),
                "analysis_method": "ESM3-based mutation effect prediction"
            }
            
        except Exception as e:
            logger.error("Mutation effect prediction failed", error=str(e))
            raise
    
    def _categorize_mutation_effect(self, stability_change: float) -> str:
        """Categorize mutation effect based on stability change"""
        if stability_change > 0.1:
            return "stabilizing"
        elif stability_change < -0.1:
            return "destabilizing"
        else:
            return "neutral"
    
    def _get_mutation_recommendation(self, stability_change: float) -> str:
        """Get recommendation based on mutation effect"""
        if stability_change > 0.2:
            return "Highly beneficial mutation - consider for optimization"
        elif stability_change > 0.05:
            return "Potentially beneficial mutation - worth testing"
        elif stability_change > -0.05:
            return "Neutral mutation - minimal impact expected"
        elif stability_change > -0.2:
            return "Potentially harmful mutation - use with caution"
        else:
            return "Likely harmful mutation - avoid unless necessary"
