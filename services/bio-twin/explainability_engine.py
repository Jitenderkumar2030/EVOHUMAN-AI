"""
Explainability Engine (XAI Layer) for AI insights and decisions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

from shared.models import (
    ExplanationData, ModelMetadata, TwinInsight,
    BioMetric, BioMetricType
)
from shared.constants import MODEL_VERSIONS, RISK_LEVELS
from shared.utils import generate_id, utc_now


class ExplainabilityEngine:
    """Provides explanations for AI insights and decisions"""
    
    def __init__(self):
        self.logger = structlog.get_logger("explainability_engine")
        
        # Biological relevance templates
        self.relevance_templates = {
            "genetic": {
                "high": "This insight is based on genetic sequence analysis with strong evidence for functional impact.",
                "medium": "This insight shows genetic variants with moderate confidence for biological effect.",
                "low": "This insight suggests potential genetic influence requiring further validation."
            },
            "physiological": {
                "high": "Strong correlation with established physiological markers and clinical evidence.",
                "medium": "Moderate physiological significance based on current biomarker data.",
                "low": "Preliminary physiological correlation requiring additional monitoring."
            },
            "cognitive": {
                "high": "High correlation with validated cognitive performance metrics and neural markers.",
                "medium": "Moderate cognitive impact supported by behavioral and neural data.",
                "low": "Suggested cognitive influence based on limited performance indicators."
            },
            "behavioral": {
                "high": "Strong behavioral pattern correlation with established psychological frameworks.",
                "medium": "Moderate behavioral significance based on activity and response patterns.",
                "low": "Emerging behavioral pattern requiring continued observation."
            }
        }
    
    def create_explanation(
        self,
        insight_content: str,
        confidence_score: float,
        model_metadata: ModelMetadata,
        contributing_factors: List[str],
        biological_context: str = "general",
        data_sources: List[str] = None,
        alternative_explanations: Optional[List[str]] = None
    ) -> ExplanationData:
        """Create comprehensive explanation for an AI insight"""
        
        # Determine risk level based on confidence and content
        risk_level = self._assess_risk_level(insight_content, confidence_score)
        
        # Generate reason based on contributing factors
        reason = self._generate_reason(insight_content, contributing_factors, confidence_score)
        
        # Generate biological relevance explanation
        biological_relevance = self._generate_biological_relevance(
            biological_context, confidence_score, data_sources or []
        )
        
        explanation = ExplanationData(
            reason=reason,
            model_metadata=model_metadata,
            confidence_score=confidence_score,
            risk_level=risk_level,
            biological_relevance=biological_relevance,
            contributing_factors=contributing_factors,
            alternative_explanations=alternative_explanations
        )
        
        self.logger.info(
            "Explanation generated",
            confidence_score=confidence_score,
            risk_level=risk_level,
            model_name=model_metadata.model_name
        )
        
        return explanation
    
    def explain_mutation_insight(
        self,
        mutation: str,
        stability_change: float,
        confidence_score: float,
        protein_context: Dict[str, Any],
        model_metadata: ModelMetadata
    ) -> ExplanationData:
        """Create explanation for protein mutation insights"""
        
        # Contributing factors for mutation analysis
        contributing_factors = [
            f"Protein sequence analysis of {len(protein_context.get('sequence', ''))} residues",
            f"Structural stability prediction (Δ{stability_change:.2f})",
            "ESM3 evolutionary language model analysis",
            f"Confidence assessment: {confidence_score:.1%}"
        ]
        
        if protein_context.get('active_site_proximity'):
            contributing_factors.append("Proximity to predicted active site")
        
        if protein_context.get('conservation_score'):
            contributing_factors.append(f"Evolutionary conservation score: {protein_context['conservation_score']:.2f}")
        
        # Generate reason
        effect_type = "stabilizing" if stability_change > 0 else "destabilizing" if stability_change < 0 else "neutral"
        reason = f"The {mutation} mutation is predicted to be {effect_type} based on protein folding analysis. "
        
        if abs(stability_change) > 1.0:
            reason += f"The predicted stability change of {stability_change:.2f} kcal/mol suggests significant structural impact. "
        else:
            reason += f"The moderate stability change of {stability_change:.2f} kcal/mol indicates subtle structural effects. "
        
        reason += "This prediction combines evolutionary context, structural modeling, and thermodynamic calculations."
        
        # Alternative explanations
        alternatives = [
            "Experimental validation may reveal different stability effects due to environmental factors",
            "Allosteric effects not captured in the structural model could modify the actual impact",
            "Post-translational modifications might alter the predicted outcome"
        ]
        
        return self.create_explanation(
            insight_content=f"Mutation {mutation} predicted {effect_type} effect",
            confidence_score=confidence_score,
            model_metadata=model_metadata,
            contributing_factors=contributing_factors,
            biological_context="genetic",
            data_sources=["ESM3", "protein_structure", "thermodynamic_model"],
            alternative_explanations=alternatives
        )
    
    def explain_biomarker_insight(
        self,
        biomarker: BioMetric,
        trend_analysis: Dict[str, Any],
        reference_ranges: Dict[str, Any],
        model_metadata: ModelMetadata
    ) -> ExplanationData:
        """Create explanation for biomarker insights"""
        
        # Determine biomarker category
        bio_context = biomarker.type.value if biomarker.type else "physiological"
        
        # Contributing factors
        contributing_factors = [
            f"Current {biomarker.name} value: {biomarker.value} {biomarker.unit}",
            f"Measurement confidence: {biomarker.confidence:.1%}",
            f"Data source: {biomarker.source}"
        ]
        
        if trend_analysis.get('trend'):
            contributing_factors.append(f"Trend analysis: {trend_analysis['trend']}")
        
        if reference_ranges:
            contributing_factors.append("Comparison with population reference ranges")
        
        # Generate reason based on biomarker value and ranges
        reason = f"The {biomarker.name} level of {biomarker.value} {biomarker.unit} "
        
        if reference_ranges:
            normal_range = reference_ranges.get('normal', {})
            if normal_range:
                low, high = normal_range.get('min', 0), normal_range.get('max', float('inf'))
                if biomarker.value < low:
                    reason += "is below the normal reference range, suggesting potential deficiency or underlying condition. "
                elif biomarker.value > high:
                    reason += "is above the normal reference range, indicating possible elevation requiring attention. "
                else:
                    reason += "falls within the normal reference range, suggesting healthy levels. "
        
        if trend_analysis.get('trend') == 'improving':
            reason += "The recent trend shows improvement, indicating positive response to interventions."
        elif trend_analysis.get('trend') == 'declining':
            reason += "The recent trend shows decline, warranting closer monitoring and potential intervention."
        
        # Confidence assessment
        confidence_score = min(biomarker.confidence, trend_analysis.get('confidence', 0.8))
        
        return self.create_explanation(
            insight_content=f"{biomarker.name} analysis reveals {trend_analysis.get('status', 'current status')}",
            confidence_score=confidence_score,
            model_metadata=model_metadata,
            contributing_factors=contributing_factors,
            biological_context=bio_context,
            data_sources=[biomarker.source, "trend_analysis", "reference_data"]
        )
    
    def explain_intervention_recommendation(
        self,
        intervention: str,
        rationale: Dict[str, Any],
        user_context: Dict[str, Any],
        confidence_score: float,
        model_metadata: ModelMetadata
    ) -> ExplanationData:
        """Create explanation for intervention recommendations"""
        
        # Contributing factors from rationale
        contributing_factors = []
        
        if rationale.get('biomarkers'):
            contributing_factors.extend([
                f"Biomarker analysis: {marker}" for marker in rationale['biomarkers'][:3]
            ])
        
        if rationale.get('goals'):
            contributing_factors.append(f"User goals alignment: {', '.join(rationale['goals'][:2])}")
        
        if rationale.get('historical_success'):
            contributing_factors.append(f"Historical success rate: {rationale['historical_success']:.1%}")
        
        if user_context.get('preferences'):
            contributing_factors.append("Personal preferences and constraints considered")
        
        # Generate comprehensive reason
        reason = f"The recommendation for '{intervention}' is based on comprehensive analysis of your current biological state. "
        
        if rationale.get('primary_driver'):
            reason += f"The primary driver is {rationale['primary_driver']}, which analysis indicates could benefit from this intervention. "
        
        if rationale.get('expected_outcome'):
            reason += f"Expected outcome: {rationale['expected_outcome']}. "
        
        if rationale.get('timeframe'):
            reason += f"Estimated timeframe for observable effects: {rationale['timeframe']}."
        
        # Alternative interventions
        alternatives = rationale.get('alternatives', [
            "Alternative lifestyle modifications could achieve similar outcomes with different approaches",
            "Combination therapies might enhance effectiveness but require careful monitoring",
            "Delayed intervention remains an option with continued observation"
        ])
        
        return self.create_explanation(
            insight_content=f"Personalized intervention recommendation: {intervention}",
            confidence_score=confidence_score,
            model_metadata=model_metadata,
            contributing_factors=contributing_factors,
            biological_context="behavioral",
            data_sources=["biomarker_analysis", "goal_optimization", "historical_data"],
            alternative_explanations=alternatives[:3]  # Limit to top 3
        )
    
    def _assess_risk_level(self, insight_content: str, confidence_score: float) -> str:
        """Assess risk level for an insight"""
        
        # High-risk keywords
        high_risk_keywords = ['critical', 'urgent', 'danger', 'severe', 'immediate', 'emergency']
        medium_risk_keywords = ['concern', 'attention', 'monitor', 'caution', 'elevated', 'abnormal']
        
        content_lower = insight_content.lower()
        
        # Check for high-risk keywords
        if any(keyword in content_lower for keyword in high_risk_keywords):
            return RISK_LEVELS["HIGH"]
        
        # Check confidence score
        if confidence_score < 0.6:
            return RISK_LEVELS["HIGH"]  # Low confidence = high risk
        
        # Check for medium-risk keywords
        if any(keyword in content_lower for keyword in medium_risk_keywords):
            return RISK_LEVELS["MEDIUM"]
        
        # Default risk assessment based on confidence
        if confidence_score > 0.8:
            return RISK_LEVELS["LOW"]
        else:
            return RISK_LEVELS["MEDIUM"]
    
    def _generate_reason(
        self, 
        insight_content: str, 
        contributing_factors: List[str], 
        confidence_score: float
    ) -> str:
        """Generate reasoning explanation"""
        
        reason = f"This insight was generated through analysis of {len(contributing_factors)} key factors: "
        
        # Summarize top contributing factors
        if contributing_factors:
            if len(contributing_factors) <= 3:
                reason += ", ".join(contributing_factors) + ". "
            else:
                reason += ", ".join(contributing_factors[:2]) + f", and {len(contributing_factors) - 2} additional factors. "
        
        # Add confidence context
        if confidence_score > 0.9:
            reason += "The high confidence score indicates strong evidence supporting this conclusion."
        elif confidence_score > 0.7:
            reason += "The moderate-to-high confidence score suggests reliable evidence for this insight."
        elif confidence_score > 0.5:
            reason += "The moderate confidence score indicates reasonable evidence, though additional validation may be beneficial."
        else:
            reason += "The lower confidence score suggests this insight should be considered preliminary and requires further validation."
        
        return reason
    
    def _generate_biological_relevance(
        self, 
        biological_context: str, 
        confidence_score: float, 
        data_sources: List[str]
    ) -> str:
        """Generate biological relevance explanation"""
        
        # Determine confidence category
        if confidence_score > 0.8:
            confidence_category = "high"
        elif confidence_score > 0.6:
            confidence_category = "medium"
        else:
            confidence_category = "low"
        
        # Get base relevance from templates
        base_relevance = self.relevance_templates.get(biological_context, {}).get(
            confidence_category, 
            "This insight shows potential biological significance based on available data."
        )
        
        # Add data source context
        if data_sources:
            base_relevance += f" Analysis incorporated data from: {', '.join(data_sources[:3])}."
            
            if len(data_sources) > 3:
                base_relevance += f" Additional {len(data_sources) - 3} data sources were also considered."
        
        return base_relevance
