# 🧬 ESM3 API Integration Guide

## Overview
The ESM3 service provides protein structure prediction, mutation analysis, and evolutionary pathway optimization using Facebook's ESM3 protein language model.

## Base URLs
- **Gateway**: `http://localhost:8000`
- **Direct ESM3 Service**: `http://localhost:8002`

## Authentication
All requests through the Gateway require Bearer token authentication:
```bash
Authorization: Bearer <your_jwt_token>
```

## 🔬 Core Endpoints

### 1. Protein Structure Analysis

**Endpoint**: `POST /esm3/analyze`

**Description**: Analyze protein sequence for structure prediction, mutation hotspots, and evolutionary potential.

**Request Body**:
```json
{
  "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
  "analysis_type": "structure_prediction",
  "include_mutations": false,
  "include_evolution": false
}
```

**Response**:
```json
{
  "sequence_id": "uuid-here",
  "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
  "sequence_length": 64,
  "predicted_structure": "High confidence structure prediction for 64 residues (confidence: 0.892)",
  "confidence_score": 0.892,
  "analysis_type": "structure_prediction",
  "contact_map": {
    "contacts": [[0.1, 0.2], [0.3, 0.4]],
    "description": "Predicted residue-residue contacts"
  },
  "mutation_analysis": {
    "hotspots": [10, 25, 67],
    "hotspot_count": 3,
    "stability_effects": "Identified 3 potential mutation sites"
  },
  "evolution_suggestion": {
    "pathways": [
      {
        "pathway_type": "stability_optimization",
        "predicted_improvement": 0.15,
        "priority": "high"
      }
    ],
    "optimization_targets": ["structural_stability", "catalytic_activity"]
  },
  "processing_time": 2.34,
  "timestamp": "2025-08-05T10:30:00Z",
  "status": "completed",
  "mock": false
}
```

### 2. Mutation Effect Prediction

**Endpoint**: `POST /esm3/predict_mutations`

**Description**: Predict the effects of specific mutations on protein stability and function.

**Request Body**:
```json
{
  "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
  "mutations": [
    {"position": 10, "from_aa": "A", "to_aa": "V"},
    {"position": 25, "from_aa": "L", "to_aa": "F"},
    {"position": 40, "from_aa": "G", "to_aa": "A"}
  ]
}
```

**Response**:
```json
{
  "mutation_effects": [
    {
      "mutation": "A11V",
      "stability_change": 0.12,
      "confidence": 0.87,
      "effect_category": "stabilizing",
      "recommendation": "Potentially beneficial mutation - worth testing"
    },
    {
      "mutation": "L26F",
      "stability_change": -0.05,
      "confidence": 0.82,
      "effect_category": "neutral",
      "recommendation": "Neutral mutation - minimal impact expected"
    }
  ],
  "wild_type_confidence": 0.89,
  "total_mutations_analyzed": 3,
  "analysis_method": "ESM3-based mutation effect prediction"
}
```

### 3. Evolutionary Pathway Analysis

**Endpoint**: `POST /esm3/evolution_analysis`

**Description**: Analyze evolutionary pathways for protein optimization based on target properties.

**Request Body**:
```json
{
  "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
  "target_properties": {
    "stability": true,
    "activity": true,
    "solubility": false,
    "binding_affinity": false
  }
}
```

**Response**:
```json
{
  "evolutionary_pathways": [
    {
      "pathway_id": "stability_optimization",
      "pathway_type": "stability",
      "target_positions": [10, 25, 67, 89],
      "suggested_mutations": ["A11V", "L26F", "G68A"],
      "predicted_improvement": 0.25,
      "confidence": 0.75,
      "description": "Stabilize 4 low-confidence regions",
      "estimated_steps": 3,
      "priority": "high",
      "ranking_score": 85.5
    }
  ],
  "baseline_confidence": 0.89,
  "optimization_potential": "medium",
  "recommendations": [
    "Focus on stabilizing low-confidence regions",
    "Test mutations individually before combining",
    "Validate predictions with experimental data"
  ]
}
```

### 4. Batch Analysis

**Endpoint**: `POST /esm3/batch_analyze`

**Description**: Analyze multiple protein sequences in batch (up to 100 sequences).

**Request Body**:
```json
{
  "sequences": [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "MKLLNVINFVFLMFVSSSKILGVNLWLRQPNLAINQENDFVLVAMKMNIRQVAQGHQETVLQMYGCNLGMTQGRQMLLKIASQAKKNNL"
  ],
  "analysis_type": "structure_prediction",
  "use_exostack": false
}
```

**Response**:
```json
{
  "batch_id": "batch-uuid-here",
  "total_sequences": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "sequence_id": "seq-0-uuid",
      "confidence_score": 0.89,
      "status": "completed"
    },
    {
      "sequence_id": "seq-1-uuid", 
      "confidence_score": 0.76,
      "status": "completed"
    }
  ]
}
```

### 5. Model Information

**Endpoint**: `GET /esm3/model_info`

**Description**: Get information about the loaded ESM3 model.

**Response**:
```json
{
  "model_name": "esm3_sm_open_v1",
  "model_path": "/app/models/esm3",
  "parameters": {
    "max_sequence_length": 1024,
    "batch_size": 8,
    "temperature": 0.7,
    "top_k": 50
  },
  "gpu_available": true,
  "model_loaded": true,
  "supported_tasks": [
    "protein_structure_prediction",
    "protein_evolution_analysis", 
    "synthetic_protein_design",
    "mutation_effect_prediction"
  ]
}
```

## 🧪 Sample Requests

### cURL Examples

**Basic Structure Analysis**:
```bash
curl -X POST "http://localhost:8000/esm3/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "analysis_type": "structure_prediction",
    "include_mutations": true,
    "include_evolution": true
  }'
```

**Mutation Analysis**:
```bash
curl -X POST "http://localhost:8000/esm3/predict_mutations" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "mutations": [
      {"position": 10, "from_aa": "A", "to_aa": "V"}
    ]
  }'
```

### Python Example

```python
import httpx
import asyncio

async def analyze_protein():
    async with httpx.AsyncClient() as client:
        # Login first
        login_response = await client.post(
            "http://localhost:8000/auth/login",
            json={"email": "user@example.com", "password": "password"}
        )
        token = login_response.json()["access_token"]
        
        # Analyze protein
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post(
            "http://localhost:8000/esm3/analyze",
            headers=headers,
            json={
                "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "analysis_type": "structure_prediction",
                "include_mutations": True,
                "include_evolution": True
            }
        )
        
        result = response.json()
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Structure: {result['predicted_structure']}")

asyncio.run(analyze_protein())
```

## 🚀 Testing the Integration

Run the comprehensive test suite:
```bash
python scripts/test_esm3_integration.py
```

## 📊 Response Formats

### Success Response
All successful responses include:
- `sequence_id`: Unique identifier for the analysis
- `confidence_score`: Model confidence (0.0 to 1.0)
- `processing_time`: Time taken in seconds
- `timestamp`: ISO format timestamp
- `status`: "completed" for successful analyses

### Error Response
```json
{
  "detail": "Error description",
  "error_code": "ESM3_ERROR",
  "timestamp": "2025-08-05T10:30:00Z"
}
```

## 🔧 Configuration

ESM3 service can be configured via `/configs/esm3.yaml`:
- Model selection (small, medium, large)
- Inference parameters
- Resource limits
- Integration settings

## 🎯 Next Steps

1. **Test the integration**: Run the test script
2. **Integrate with Bio-Twin**: Connect to bio-digital twin engine
3. **Setup ExoStack**: Enable distributed processing
4. **Add SymbioticAIS**: Implement feedback loops
5. **Build UI**: Create protein analysis dashboard

## 📚 Additional Resources

- [ESM3 Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- [Facebook ESM Repository](https://github.com/facebookresearch/esm)
- [EvoHuman.AI Documentation](../README.md)
