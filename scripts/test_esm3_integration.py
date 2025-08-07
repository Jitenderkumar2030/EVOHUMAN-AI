#!/usr/bin/env python3
"""
ESM3 Integration Test Script for EvoHuman.AI
Tests the complete ESM3 service integration
"""
import asyncio
import httpx
import json
import time
from typing import Dict, Any, List


# Test configuration
GATEWAY_URL = "http://localhost:8000"
ESM3_SERVICE_URL = "http://localhost:8002"
TEST_SEQUENCE = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
TEST_USER_EMAIL = "test@evohuman.ai"
TEST_USER_PASSWORD = "testpass123"


class ESM3IntegrationTester:
    """Test ESM3 service integration"""
    
    def __init__(self):
        self.gateway_client = httpx.AsyncClient(base_url=GATEWAY_URL, timeout=60.0)
        self.esm3_client = httpx.AsyncClient(base_url=ESM3_SERVICE_URL, timeout=60.0)
        self.auth_token = None
    
    async def setup(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        
        # Try to register test user (ignore if already exists)
        try:
            register_data = {
                "email": TEST_USER_EMAIL,
                "username": "testuser",
                "password": TEST_USER_PASSWORD,
                "full_name": "Test User"
            }
            response = await self.gateway_client.post("/auth/register", json=register_data)
            if response.status_code == 200:
                print("✅ Test user registered")
            else:
                print("ℹ️ Test user already exists")
        except Exception as e:
            print(f"ℹ️ User registration: {e}")
        
        # Login to get auth token
        try:
            login_data = {
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD
            }
            response = await self.gateway_client.post("/auth/login", json=login_data)
            if response.status_code == 200:
                result = response.json()
                self.auth_token = result["access_token"]
                self.gateway_client.headers["Authorization"] = f"Bearer {self.auth_token}"
                print("✅ Authentication successful")
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
        
        return True
    
    async def test_health_checks(self):
        """Test health check endpoints"""
        print("\n🏥 Testing health checks...")
        
        # Test Gateway health
        try:
            response = await self.gateway_client.get("/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Gateway health: {health['status']}")
            else:
                print(f"❌ Gateway health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Gateway health error: {e}")
        
        # Test ESM3 service health
        try:
            response = await self.esm3_client.get("/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ ESM3 service health: {health['status']}")
                
                # Print dependency status
                deps = health.get("dependencies", {})
                for dep, status in deps.items():
                    status_icon = "✅" if status else "❌"
                    print(f"  {status_icon} {dep}: {'OK' if status else 'FAIL'}")
            else:
                print(f"❌ ESM3 health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ ESM3 health error: {e}")
    
    async def test_model_info(self):
        """Test model information endpoint"""
        print("\n📊 Testing model info...")
        
        try:
            # Test via Gateway
            response = await self.gateway_client.get("/esm3/model_info")
            if response.status_code == 200:
                info = response.json()
                print("✅ Model info retrieved via Gateway:")
                print(f"  Model: {info.get('model_name', 'Unknown')}")
                print(f"  GPU Available: {info.get('gpu_available', False)}")
                print(f"  Model Loaded: {info.get('model_loaded', False)}")
                
                if info.get('mock', False):
                    print("  ⚠️ Running in mock mode")
            else:
                print(f"❌ Model info failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Model info error: {e}")
    
    async def test_protein_analysis(self):
        """Test basic protein analysis"""
        print("\n🧬 Testing protein analysis...")
        
        try:
            payload = {
                "sequence": TEST_SEQUENCE,
                "analysis_type": "structure_prediction",
                "include_mutations": False,
                "include_evolution": False
            }
            
            start_time = time.time()
            response = await self.gateway_client.post("/esm3/analyze", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Protein analysis successful:")
                print(f"  Sequence ID: {result.get('sequence_id', 'N/A')}")
                print(f"  Confidence: {result.get('confidence_score', 0.0):.3f}")
                print(f"  Processing time: {end_time - start_time:.2f}s")
                print(f"  Structure: {result.get('predicted_structure', 'N/A')[:50]}...")
                
                if result.get('mock', False):
                    print("  ⚠️ Mock analysis result")
                
                return result
            else:
                print(f"❌ Protein analysis failed: {response.status_code}")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"❌ Protein analysis error: {e}")
        
        return None
    
    async def test_mutation_analysis(self):
        """Test mutation effect prediction"""
        print("\n🔬 Testing mutation analysis...")
        
        try:
            mutations = [
                {"position": 10, "from_aa": "A", "to_aa": "V"},
                {"position": 25, "from_aa": "L", "to_aa": "F"},
                {"position": 40, "from_aa": "G", "to_aa": "A"}
            ]
            
            payload = {
                "sequence": TEST_SEQUENCE,
                "mutations": mutations
            }
            
            response = await self.gateway_client.post("/esm3/predict_mutations", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Mutation analysis successful:")
                print(f"  Mutations analyzed: {result.get('total_mutations_analyzed', 0)}")
                
                effects = result.get('mutation_effects', [])
                for effect in effects[:3]:  # Show first 3
                    if 'error' not in effect:
                        print(f"  {effect.get('mutation', 'N/A')}: "
                              f"{effect.get('stability_change', 0.0):+.3f} "
                              f"({effect.get('effect_category', 'unknown')})")
                
                if result.get('mock', False):
                    print("  ⚠️ Mock mutation analysis")
                
                return result
            else:
                print(f"❌ Mutation analysis failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Mutation analysis error: {e}")
        
        return None
    
    async def test_evolution_analysis(self):
        """Test evolutionary pathway analysis"""
        print("\n🧬 Testing evolution analysis...")
        
        try:
            target_properties = {
                "stability": True,
                "activity": True,
                "solubility": False
            }
            
            payload = {
                "sequence": TEST_SEQUENCE,
                "target_properties": target_properties
            }
            
            response = await self.gateway_client.post("/esm3/evolution_analysis", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Evolution analysis successful:")
                print(f"  Baseline confidence: {result.get('baseline_confidence', 0.0):.3f}")
                print(f"  Optimization potential: {result.get('optimization_potential', 'unknown')}")
                
                pathways = result.get('evolutionary_pathways', [])
                print(f"  Pathways found: {len(pathways)}")
                
                for i, pathway in enumerate(pathways[:2]):  # Show first 2
                    print(f"    {i+1}. {pathway.get('pathway_type', 'unknown')}: "
                          f"{pathway.get('predicted_improvement', 0.0):.3f} improvement")
                
                recommendations = result.get('recommendations', [])
                if recommendations:
                    print(f"  Top recommendation: {recommendations[0]}")
                
                if result.get('mock', False):
                    print("  ⚠️ Mock evolution analysis")
                
                return result
            else:
                print(f"❌ Evolution analysis failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Evolution analysis error: {e}")
        
        return None
    
    async def test_batch_analysis(self):
        """Test batch protein analysis"""
        print("\n📦 Testing batch analysis...")
        
        try:
            sequences = [
                TEST_SEQUENCE,
                "MKLLNVINFVFLMFVSSSKILGVNLWLRQPNLAINQENDFVLVAMKMNIRQVAQGHQETVLQMYGCNLGMTQGRQMLLKIASQAKKNNL",
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
            ]
            
            payload = {
                "sequences": sequences,
                "analysis_type": "structure_prediction",
                "use_exostack": False
            }
            
            response = await self.gateway_client.post("/esm3/batch_analyze", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Batch analysis successful:")
                print(f"  Total sequences: {result.get('total_sequences', 0)}")
                print(f"  Successful: {result.get('successful', 0)}")
                print(f"  Failed: {result.get('failed', 0)}")
                
                return result
            else:
                print(f"❌ Batch analysis failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Batch analysis error: {e}")
        
        return None
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🧬 ESM3 Integration Test Suite")
        print("=" * 50)
        
        # Setup
        if not await self.setup():
            print("❌ Setup failed, aborting tests")
            return
        
        # Run tests
        await self.test_health_checks()
        await self.test_model_info()
        await self.test_protein_analysis()
        await self.test_mutation_analysis()
        await self.test_evolution_analysis()
        await self.test_batch_analysis()
        
        print("\n" + "=" * 50)
        print("🎉 ESM3 Integration tests completed!")
        print("\nNext steps:")
        print("1. Check logs for any errors")
        print("2. Test with real ESM3 model (if not in mock mode)")
        print("3. Integrate with Bio-Twin engine")
        print("4. Setup ExoStack distributed processing")
    
    async def cleanup(self):
        """Cleanup test resources"""
        await self.gateway_client.aclose()
        await self.esm3_client.aclose()


async def main():
    """Main test function"""
    tester = ESM3IntegrationTester()
    try:
        await tester.run_all_tests()
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
