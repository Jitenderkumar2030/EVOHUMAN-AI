"""
Integration Tests for EvoHuman.AI Services
Tests end-to-end workflows and service interactions
"""
import pytest
import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List
import structlog
from dataclasses import dataclass


logger = structlog.get_logger("integration-tests")


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    base_url: str
    health_path: str = "/health"
    timeout: int = 30


class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.services = [
            ServiceEndpoint("frontend", "http://localhost:3000", "/"),
            ServiceEndpoint("api-gateway", "http://localhost:8000"),
            ServiceEndpoint("aice-service", "http://localhost:8001"),
            ServiceEndpoint("proteus-service", "http://localhost:8002"),
            ServiceEndpoint("esm3-service", "http://localhost:8003"),
            ServiceEndpoint("symbiotic-service", "http://localhost:8004"),
            ServiceEndpoint("bio-twin-service", "http://localhost:8005"),
            ServiceEndpoint("exostack-service", "http://localhost:8006"),
        ]
        
        self.test_user_id = "integration_test_user"
        self.session: aiohttp.ClientSession = None
    
    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        # Wait for all services to be ready
        await self._wait_for_services()
        
        # Setup test data
        await self._setup_test_data()
        
        logger.info("Integration test setup complete")
    
    async def teardown(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
        
        # Cleanup test data
        await self._cleanup_test_data()
        
        logger.info("Integration test teardown complete")
    
    async def _wait_for_services(self):
        """Wait for all services to be healthy"""
        logger.info("Waiting for services to be ready...")
        
        for service in self.services:
            retries = 30
            while retries > 0:
                try:
                    async with self.session.get(
                        f"{service.base_url}{service.health_path}"
                    ) as response:
                        if response.status == 200:
                            logger.info(f"✅ {service.name} is ready")
                            break
                except Exception:
                    pass
                
                retries -= 1
                if retries > 0:
                    await asyncio.sleep(2)
                else:
                    logger.warning(f"⚠️  {service.name} not ready, continuing tests")
    
    async def _setup_test_data(self):
        """Setup test data across services"""
        try:
            # Create test user in AiCE service
            await self.session.post(
                "http://localhost:8001/users",
                json={
                    "id": self.test_user_id,
                    "name": "Integration Test User",
                    "email": "integration@test.com",
                    "biological_age": 28,
                    "chronological_age": 32,
                }
            )
            
            # Initialize bio-twin
            await self.session.post(
                "http://localhost:8005/bio-twin/initialize",
                json={
                    "user_id": self.test_user_id,
                    "initial_metrics": {
                        "health_score": 85,
                        "energy_level": 78,
                        "cognitive_index": 92,
                    }
                }
            )
            
            # Initialize multi-agent system
            await self.session.post(
                "http://localhost:8004/multi_agent/initialize",
                json={"user_id": self.test_user_id}
            )
            
            logger.info("Test data setup complete")
            
        except Exception as e:
            logger.warning(f"Test data setup failed: {e}, using mock data")
    
    async def _cleanup_test_data(self):
        """Cleanup test data"""
        try:
            # Delete test user data
            await self.session.delete(f"http://localhost:8001/users/{self.test_user_id}")
            await self.session.delete(f"http://localhost:8005/bio-twin/{self.test_user_id}")
            
            logger.info("Test data cleanup complete")
        except Exception as e:
            logger.warning(f"Test data cleanup failed: {e}")


@pytest.fixture
async def integration_suite():
    """Integration test suite fixture"""
    suite = IntegrationTestSuite()
    await suite.setup()
    yield suite
    await suite.teardown()


class TestServiceHealth:
    """Test service health and availability"""
    
    async def test_all_services_healthy(self, integration_suite):
        """Test that all services are healthy"""
        for service in integration_suite.services:
            async with integration_suite.session.get(
                f"{service.base_url}{service.health_path}"
            ) as response:
                assert response.status == 200, f"{service.name} is not healthy"
                
                if service.name != "frontend":
                    data = await response.json()
                    assert data.get("status") == "healthy", f"{service.name} reports unhealthy status"
    
    async def test_service_metrics_available(self, integration_suite):
        """Test that service metrics are available"""
        metric_services = [
            "aice-service", "proteus-service", "esm3-service", 
            "symbiotic-service", "bio-twin-service", "exostack-service"
        ]
        
        for service_name in metric_services:
            service = next(s for s in integration_suite.services if s.name == service_name)
            
            async with integration_suite.session.get(
                f"{service.base_url}/metrics"
            ) as response:
                assert response.status == 200, f"{service_name} metrics not available"
                
                metrics_text = await response.text()
                assert "http_requests_total" in metrics_text, f"{service_name} missing request metrics"


class TestBioTwinWorkflow:
    """Test complete bio-twin workflow"""
    
    async def test_bio_twin_data_retrieval(self, integration_suite):
        """Test bio-twin data retrieval"""
        async with integration_suite.session.get(
            f"http://localhost:8005/bio-twin/{integration_suite.test_user_id}"
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert "current_metrics" in data
            assert "timeline" in data
            assert data["current_metrics"]["health_score"] > 0
    
    async def test_bio_twin_metrics_update(self, integration_suite):
        """Test bio-twin metrics update"""
        update_data = {
            "health_score": 88,
            "energy_level": 82,
            "stress_level": 20,
        }
        
        async with integration_suite.session.post(
            f"http://localhost:8005/bio-twin/{integration_suite.test_user_id}/update",
            json=update_data
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert data["updated"] == True
    
    async def test_cognitive_assessment_workflow(self, integration_suite):
        """Test cognitive assessment workflow"""
        # Start assessment
        async with integration_suite.session.post(
            f"http://localhost:8001/cognitive/assess",
            json={
                "user_id": integration_suite.test_user_id,
                "assessment_type": "comprehensive"
            }
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assessment_id = data["assessment_id"]
        
        # Get assessment results
        async with integration_suite.session.get(
            f"http://localhost:8001/cognitive/assessment/{assessment_id}"
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert "cognitive_scores" in data
            assert len(data["cognitive_scores"]) > 0


class TestProteinAnalysisWorkflow:
    """Test protein analysis workflow"""
    
    async def test_protein_sequence_analysis(self, integration_suite):
        """Test protein sequence analysis"""
        test_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        
        # Submit analysis request
        async with integration_suite.session.post(
            "http://localhost:8003/analyze_protein",
            json={
                "sequence": test_sequence,
                "analysis_type": "structure_prediction",
                "include_mutations": True,
                "include_evolution": True,
            }
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert "analysis_id" in data
            assert "confidence_score" in data
            assert data["sequence_length"] == len(test_sequence)
    
    async def test_batch_protein_analysis(self, integration_suite):
        """Test batch protein analysis"""
        sequences = [
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
        ]
        
        async with integration_suite.session.post(
            "http://localhost:8003/batch_analyze",
            json={
                "sequences": sequences,
                "analysis_type": "structure_prediction"
            }
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert "batch_id" in data
            assert len(data["results"]) == len(sequences)


class TestCellularSimulationWorkflow:
    """Test cellular simulation workflow"""
    
    async def test_cellular_automata_simulation(self, integration_suite):
        """Test cellular automata simulation"""
        # Start simulation
        async with integration_suite.session.post(
            "http://localhost:8002/simulate/cellular_automata",
            json={
                "tissue_type": "neural",
                "initial_cell_count": 1000,
                "simulation_steps": 50,
            }
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert "simulation_id" in data
            assert data["status"] in ["running", "completed"]
    
    async def test_regeneration_simulation(self, integration_suite):
        """Test tissue regeneration simulation"""
        async with integration_suite.session.post(
            "http://localhost:8002/simulate/wound_healing",
            json={
                "wound_type": "acute",
                "wound_size": [10, 10, 5],
                "simulation_days": 14,
                "patient_age": 35,
            }
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert "simulation_id" in data
            assert "healing_phases" in data


class TestMultiAgentSystemWorkflow:
    """Test multi-agent system workflow"""
    
    async def test_agent_system_initialization(self, integration_suite):
        """Test multi-agent system initialization"""
        async with integration_suite.session.get(
            f"http://localhost:8004/multi_agent/status/{integration_suite.test_user_id}"
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert data["status"] == "active"
            assert "system_status" in data
            assert data["system_status"]["agent_count"] > 0
    
    async def test_agent_step_execution(self, integration_suite):
        """Test agent step execution"""
        human_input = {
            "satisfaction": 0.8,
            "goals": ["improve_health", "increase_longevity"],
            "feedback": {"energy_level": 7, "mood": 8}
        }
        
        async with integration_suite.session.post(
            "http://localhost:8004/multi_agent/step",
            json={
                "user_id": integration_suite.test_user_id,
                "human_input": human_input
            }
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert "step_results" in data
            assert data["step_results"]["actions_taken"] >= 0
    
    async def test_human_feedback_processing(self, integration_suite):
        """Test human feedback processing"""
        feedback_data = {
            "satisfaction": 0.7,
            "goals": ["better_sleep", "stress_reduction"],
            "feedback": {"sleep_quality": 6, "stress_level": 4}
        }
        
        async with integration_suite.session.post(
            "http://localhost:8004/multi_agent/human_feedback",
            json={
                "user_id": integration_suite.test_user_id,
                "feedback_data": feedback_data
            }
        ) as response:
            assert response.status == 200
            
            data = await response.json()
            assert data["feedback_processed"] == True
            assert "symbiotic_response" in data


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    async def test_complete_user_journey(self, integration_suite):
        """Test complete user journey from registration to insights"""
        user_id = f"e2e_user_{int(time.time())}"
        
        try:
            # 1. User registration (simulated)
            async with integration_suite.session.post(
                "http://localhost:8001/users",
                json={
                    "id": user_id,
                    "name": "E2E Test User",
                    "email": f"{user_id}@test.com",
                    "biological_age": 30,
                    "chronological_age": 35,
                }
            ) as response:
                assert response.status == 200
            
            # 2. Bio-twin initialization
            async with integration_suite.session.post(
                "http://localhost:8005/bio-twin/initialize",
                json={
                    "user_id": user_id,
                    "initial_metrics": {
                        "health_score": 80,
                        "energy_level": 75,
                        "cognitive_index": 88,
                    }
                }
            ) as response:
                assert response.status == 200
            
            # 3. Cognitive assessment
            async with integration_suite.session.post(
                f"http://localhost:8001/cognitive/assess",
                json={
                    "user_id": user_id,
                    "assessment_type": "comprehensive"
                }
            ) as response:
                assert response.status == 200
            
            # 4. Protein analysis
            async with integration_suite.session.post(
                "http://localhost:8003/analyze_protein",
                json={
                    "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
                    "analysis_type": "structure_prediction"
                }
            ) as response:
                assert response.status == 200
            
            # 5. Multi-agent system interaction
            async with integration_suite.session.post(
                "http://localhost:8004/multi_agent/initialize",
                json={"user_id": user_id}
            ) as response:
                assert response.status == 200
            
            # 6. Get comprehensive insights
            async with integration_suite.session.get(
                f"http://localhost:8005/bio-twin/{user_id}"
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "current_metrics" in data
                assert "ai_insights" in data
            
            logger.info("Complete user journey test passed", user_id=user_id)
            
        finally:
            # Cleanup
            try:
                await integration_suite.session.delete(f"http://localhost:8001/users/{user_id}")
                await integration_suite.session.delete(f"http://localhost:8005/bio-twin/{user_id}")
            except Exception:
                pass
    
    async def test_service_resilience(self, integration_suite):
        """Test system resilience under load"""
        # Simulate concurrent requests
        tasks = []
        
        for i in range(10):
            # Bio-twin requests
            task = integration_suite.session.get(
                f"http://localhost:8005/bio-twin/{integration_suite.test_user_id}"
            )
            tasks.append(task)
            
            # Protein analysis requests
            task = integration_suite.session.post(
                "http://localhost:8003/analyze_protein",
                json={
                    "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
                    "analysis_type": "structure_prediction"
                }
            )
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most requests succeeded
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        success_rate = len(successful_responses) / len(responses)
        
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"
        
        logger.info(f"Service resilience test passed with {success_rate:.2%} success rate")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
