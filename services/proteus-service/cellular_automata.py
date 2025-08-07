"""
Proteus Cellular Automata Engine
Advanced cellular behavior simulation for biological modeling
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
import structlog
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist


logger = structlog.get_logger("cellular-automata")


class CellType(Enum):
    STEM = "stem"
    NEURAL = "neural"
    CARDIAC = "cardiac"
    HEPATIC = "hepatic"
    MUSCLE = "muscle"
    EPITHELIAL = "epithelial"
    IMMUNE = "immune"
    DEAD = "dead"


class CellState(Enum):
    QUIESCENT = "quiescent"
    PROLIFERATING = "proliferating"
    DIFFERENTIATING = "differentiating"
    APOPTOTIC = "apoptotic"
    SENESCENT = "senescent"


@dataclass
class Cell:
    """Individual cell in the simulation"""
    id: str
    cell_type: CellType
    state: CellState
    position: Tuple[int, int, int]  # 3D coordinates
    age: int  # Cell age in simulation steps
    energy: float  # Energy level (0-1)
    health: float  # Health status (0-1)
    division_count: int  # Number of times divided
    proteins: Dict[str, float]  # Protein concentrations
    metabolites: Dict[str, float]  # Metabolite concentrations
    signaling_molecules: Dict[str, float]  # Signaling molecule concentrations
    genetic_factors: Dict[str, float]  # Gene expression levels
    neighbors: Set[str] = field(default_factory=set)  # Neighboring cell IDs
    
    def __post_init__(self):
        # Initialize default molecular concentrations
        if not self.proteins:
            self.proteins = self._initialize_proteins()
        if not self.metabolites:
            self.metabolites = self._initialize_metabolites()
        if not self.signaling_molecules:
            self.signaling_molecules = self._initialize_signaling()
        if not self.genetic_factors:
            self.genetic_factors = self._initialize_genetics()
    
    def _initialize_proteins(self) -> Dict[str, float]:
        """Initialize protein concentrations based on cell type"""
        base_proteins = {
            "p53": 0.5,  # Tumor suppressor
            "cyclin_d1": 0.3,  # Cell cycle regulation
            "bcl2": 0.4,  # Apoptosis regulation
            "vegf": 0.2,  # Vascular growth factor
            "tgf_beta": 0.3,  # Growth factor
            "nf_kb": 0.4,  # Transcription factor
            "akt": 0.5,  # Cell survival
            "mtor": 0.3,  # Growth regulation
        }
        
        # Modify based on cell type
        if self.cell_type == CellType.STEM:
            base_proteins["oct4"] = 0.8
            base_proteins["nanog"] = 0.7
            base_proteins["sox2"] = 0.8
        elif self.cell_type == CellType.NEURAL:
            base_proteins["neurofilament"] = 0.9
            base_proteins["synaptophysin"] = 0.7
            base_proteins["gfap"] = 0.6
        elif self.cell_type == CellType.CARDIAC:
            base_proteins["troponin"] = 0.9
            base_proteins["myosin"] = 0.8
            base_proteins["actin"] = 0.8
        
        return base_proteins
    
    def _initialize_metabolites(self) -> Dict[str, float]:
        """Initialize metabolite concentrations"""
        return {
            "glucose": 0.8,
            "atp": 0.7,
            "lactate": 0.2,
            "oxygen": 0.9,
            "co2": 0.3,
            "calcium": 0.1,
            "sodium": 0.4,
            "potassium": 0.6
        }
    
    def _initialize_signaling(self) -> Dict[str, float]:
        """Initialize signaling molecule concentrations"""
        return {
            "growth_factors": 0.3,
            "cytokines": 0.2,
            "hormones": 0.4,
            "neurotransmitters": 0.1 if self.cell_type == CellType.NEURAL else 0.0,
            "chemokines": 0.2
        }
    
    def _initialize_genetics(self) -> Dict[str, float]:
        """Initialize gene expression levels"""
        return {
            "cell_cycle_genes": 0.5,
            "apoptosis_genes": 0.3,
            "differentiation_genes": 0.4,
            "stress_response_genes": 0.2,
            "repair_genes": 0.6,
            "metabolism_genes": 0.7
        }


@dataclass
class TissueEnvironment:
    """Tissue environment parameters"""
    oxygen_level: float = 0.9
    nutrient_level: float = 0.8
    ph_level: float = 7.4
    temperature: float = 37.0  # Celsius
    mechanical_stress: float = 0.1
    toxin_level: float = 0.0
    inflammation_level: float = 0.1
    vascularization: float = 0.7


class CellularAutomata:
    """Advanced cellular automata for biological simulation"""
    
    def __init__(self, grid_size: Tuple[int, int, int] = (100, 100, 20)):
        self.grid_size = grid_size
        self.cells: Dict[str, Cell] = {}
        self.grid = np.zeros(grid_size, dtype=object)  # 3D grid
        self.environment = TissueEnvironment()
        self.simulation_step = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Simulation parameters
        self.division_probability = 0.1
        self.death_probability = 0.01
        self.differentiation_probability = 0.05
        self.migration_probability = 0.2
        
        # Molecular diffusion parameters
        self.diffusion_coefficients = {
            "oxygen": 0.8,
            "glucose": 0.6,
            "growth_factors": 0.4,
            "cytokines": 0.5,
            "waste_products": 0.7
        }
        
        logger.info("Cellular Automata initialized", grid_size=grid_size)
    
    async def initialize_tissue(
        self,
        tissue_type: str,
        initial_cell_count: int = 1000,
        cell_type_distribution: Dict[CellType, float] = None
    ):
        """Initialize tissue with specified cell types"""
        
        if cell_type_distribution is None:
            cell_type_distribution = {
                CellType.STEM: 0.1,
                CellType.NEURAL: 0.3,
                CellType.CARDIAC: 0.2,
                CellType.HEPATIC: 0.2,
                CellType.MUSCLE: 0.1,
                CellType.EPITHELIAL: 0.1
            }
        
        logger.info("Initializing tissue", 
                   tissue_type=tissue_type,
                   cell_count=initial_cell_count)
        
        # Clear existing cells
        self.cells.clear()
        self.grid.fill(None)
        
        # Place initial cells
        for i in range(initial_cell_count):
            # Determine cell type based on distribution
            cell_type = self._sample_cell_type(cell_type_distribution)
            
            # Find random position
            position = self._find_empty_position()
            if position is None:
                break  # Grid full
            
            # Create cell
            cell = Cell(
                id=f"cell_{i}",
                cell_type=cell_type,
                state=CellState.QUIESCENT,
                position=position,
                age=0,
                energy=np.random.uniform(0.7, 1.0),
                health=np.random.uniform(0.8, 1.0),
                division_count=0
            )
            
            # Place in grid and register
            self.cells[cell.id] = cell
            self.grid[position] = cell.id
        
        # Update neighborhood relationships
        await self._update_neighborhoods()
        
        logger.info("Tissue initialized", 
                   actual_cells=len(self.cells),
                   cell_types={ct.value: sum(1 for c in self.cells.values() if c.cell_type == ct) 
                              for ct in CellType})
    
    async def simulate_steps(self, num_steps: int) -> Dict[str, Any]:
        """Run cellular automata simulation for specified steps"""
        
        logger.info("Starting simulation", num_steps=num_steps)
        
        simulation_data = {
            "initial_cells": len(self.cells),
            "steps_completed": 0,
            "cell_counts_over_time": [],
            "events": []
        }
        
        try:
            for step in range(num_steps):
                self.simulation_step += 1
                
                # Run simulation step
                step_events = await self._simulate_single_step()
                
                # Record data
                cell_counts = {ct.value: sum(1 for c in self.cells.values() if c.cell_type == ct) 
                              for ct in CellType}
                simulation_data["cell_counts_over_time"].append(cell_counts)
                simulation_data["events"].extend(step_events)
                
                # Log progress
                if step % 10 == 0:
                    logger.info("Simulation progress", 
                               step=step,
                               total_cells=len(self.cells),
                               events_this_step=len(step_events))
            
            simulation_data["steps_completed"] = num_steps
            simulation_data["final_cells"] = len(self.cells)
            
            logger.info("Simulation completed", 
                       steps=num_steps,
                       final_cells=len(self.cells))
            
            return simulation_data
            
        except Exception as e:
            logger.error("Simulation failed", error=str(e))
            raise
    
    async def _simulate_single_step(self) -> List[Dict[str, Any]]:
        """Simulate a single time step"""
        
        events = []
        
        # Update molecular concentrations (diffusion)
        await self._update_molecular_diffusion()
        
        # Process each cell
        cells_to_process = list(self.cells.values())
        
        for cell in cells_to_process:
            # Skip if cell was removed during this step
            if cell.id not in self.cells:
                continue
            
            # Update cell state
            cell_events = await self._update_cell_state(cell)
            events.extend(cell_events)
            
            # Cell division
            if await self._should_cell_divide(cell):
                division_event = await self._divide_cell(cell)
                if division_event:
                    events.append(division_event)
            
            # Cell death
            elif await self._should_cell_die(cell):
                death_event = await self._kill_cell(cell)
                if death_event:
                    events.append(death_event)
            
            # Cell differentiation
            elif await self._should_cell_differentiate(cell):
                diff_event = await self._differentiate_cell(cell)
                if diff_event:
                    events.append(diff_event)
            
            # Cell migration
            elif await self._should_cell_migrate(cell):
                migration_event = await self._migrate_cell(cell)
                if migration_event:
                    events.append(migration_event)
        
        # Update neighborhoods after all changes
        await self._update_neighborhoods()
        
        return events
    
    async def _update_cell_state(self, cell: Cell) -> List[Dict[str, Any]]:
        """Update individual cell state"""
        
        events = []
        
        # Age the cell
        cell.age += 1
        
        # Update energy based on environment and metabolism
        energy_change = self._calculate_energy_change(cell)
        cell.energy = max(0.0, min(1.0, cell.energy + energy_change))
        
        # Update health based on various factors
        health_change = self._calculate_health_change(cell)
        cell.health = max(0.0, min(1.0, cell.health + health_change))
        
        # Update molecular concentrations
        await self._update_cell_molecules(cell)
        
        # Check for state transitions
        new_state = self._determine_cell_state(cell)
        if new_state != cell.state:
            events.append({
                "type": "state_change",
                "cell_id": cell.id,
                "old_state": cell.state.value,
                "new_state": new_state.value,
                "step": self.simulation_step
            })
            cell.state = new_state
        
        return events
    
    def _calculate_energy_change(self, cell: Cell) -> float:
        """Calculate energy change for a cell"""
        
        # Base metabolism cost
        energy_cost = -0.02
        
        # Environmental factors
        if self.environment.oxygen_level < 0.5:
            energy_cost -= 0.05  # Hypoxia
        
        if self.environment.nutrient_level < 0.5:
            energy_cost -= 0.03  # Nutrient depletion
        
        # Cell state factors
        if cell.state == CellState.PROLIFERATING:
            energy_cost -= 0.08  # Division is expensive
        elif cell.state == CellState.DIFFERENTIATING:
            energy_cost -= 0.05  # Differentiation costs energy
        
        # Neighbor effects (crowding)
        if len(cell.neighbors) > 6:
            energy_cost -= 0.02  # Overcrowding stress
        
        # Add some randomness
        energy_cost += np.random.normal(0, 0.01)
        
        return energy_cost
    
    def _calculate_health_change(self, cell: Cell) -> float:
        """Calculate health change for a cell"""
        
        health_change = 0.0
        
        # Age-related decline
        if cell.age > 100:
            health_change -= 0.001 * (cell.age - 100)
        
        # Energy-related health
        if cell.energy < 0.3:
            health_change -= 0.02
        elif cell.energy > 0.8:
            health_change += 0.005
        
        # Environmental stress
        if self.environment.toxin_level > 0.1:
            health_change -= self.environment.toxin_level * 0.05
        
        if self.environment.inflammation_level > 0.2:
            health_change -= self.environment.inflammation_level * 0.03
        
        # Division stress
        if cell.division_count > 10:
            health_change -= 0.001 * cell.division_count
        
        return health_change
    
    async def _update_cell_molecules(self, cell: Cell):
        """Update molecular concentrations in cell"""
        
        # Simplified molecular dynamics
        
        # Protein synthesis and degradation
        for protein, concentration in cell.proteins.items():
            synthesis_rate = cell.genetic_factors.get(f"{protein}_gene", 0.5) * 0.1
            degradation_rate = concentration * 0.05
            
            new_concentration = concentration + synthesis_rate - degradation_rate
            cell.proteins[protein] = max(0.0, min(1.0, new_concentration))
        
        # Metabolite consumption and production
        if cell.state == CellState.PROLIFERATING:
            cell.metabolites["glucose"] = max(0.0, cell.metabolites["glucose"] - 0.1)
            cell.metabolites["atp"] = max(0.0, cell.metabolites["atp"] - 0.05)
        
        # Oxygen consumption
        cell.metabolites["oxygen"] = max(0.0, 
            cell.metabolites["oxygen"] - 0.02 + self.environment.oxygen_level * 0.01)
    
    def _determine_cell_state(self, cell: Cell) -> CellState:
        """Determine cell state based on current conditions"""
        
        # Death conditions
        if cell.health < 0.1 or cell.energy < 0.05:
            return CellState.APOPTOTIC
        
        # Senescence conditions
        if cell.age > 200 or cell.division_count > 20:
            return CellState.SENESCENT
        
        # Proliferation conditions
        if (cell.energy > 0.7 and cell.health > 0.6 and 
            len(cell.neighbors) < 4 and cell.proteins.get("cyclin_d1", 0) > 0.6):
            return CellState.PROLIFERATING
        
        # Differentiation conditions
        if (cell.cell_type == CellType.STEM and cell.age > 50 and 
            cell.signaling_molecules.get("growth_factors", 0) > 0.5):
            return CellState.DIFFERENTIATING
        
        # Default to quiescent
        return CellState.QUIESCENT
    
    async def _should_cell_divide(self, cell: Cell) -> bool:
        """Determine if cell should divide"""
        
        if cell.state != CellState.PROLIFERATING:
            return False
        
        if cell.energy < 0.6 or cell.health < 0.5:
            return False
        
        if len(cell.neighbors) >= 6:  # Too crowded
            return False
        
        # Probabilistic division
        division_prob = self.division_probability
        
        # Modify based on cell type
        if cell.cell_type == CellType.STEM:
            division_prob *= 1.5
        elif cell.cell_type in [CellType.NEURAL, CellType.CARDIAC]:
            division_prob *= 0.3  # These cells divide less
        
        return np.random.random() < division_prob
    
    async def _divide_cell(self, parent_cell: Cell) -> Optional[Dict[str, Any]]:
        """Divide a cell into two daughter cells"""
        
        # Find position for daughter cell
        daughter_position = self._find_adjacent_empty_position(parent_cell.position)
        if daughter_position is None:
            return None  # No space for division
        
        # Create daughter cell
        daughter_id = f"cell_{len(self.cells)}_{self.simulation_step}"
        daughter_cell = Cell(
            id=daughter_id,
            cell_type=parent_cell.cell_type,
            state=CellState.QUIESCENT,
            position=daughter_position,
            age=0,
            energy=parent_cell.energy * 0.6,  # Energy split
            health=parent_cell.health * 0.9,
            division_count=0,
            proteins=parent_cell.proteins.copy(),
            metabolites=parent_cell.metabolites.copy(),
            signaling_molecules=parent_cell.signaling_molecules.copy(),
            genetic_factors=parent_cell.genetic_factors.copy()
        )
        
        # Update parent cell
        parent_cell.energy *= 0.4  # Energy cost of division
        parent_cell.division_count += 1
        parent_cell.state = CellState.QUIESCENT
        
        # Add daughter to simulation
        self.cells[daughter_id] = daughter_cell
        self.grid[daughter_position] = daughter_id
        
        return {
            "type": "cell_division",
            "parent_id": parent_cell.id,
            "daughter_id": daughter_id,
            "step": self.simulation_step
        }
    
    async def _should_cell_die(self, cell: Cell) -> bool:
        """Determine if cell should die"""
        
        # Definite death conditions
        if cell.state == CellState.APOPTOTIC:
            return True
        
        if cell.health < 0.1 or cell.energy < 0.05:
            return True
        
        # Age-related death
        if cell.age > 300:
            return np.random.random() < 0.5
        
        # Random death
        death_prob = self.death_probability
        
        # Modify based on conditions
        if cell.health < 0.3:
            death_prob *= 5
        
        if self.environment.toxin_level > 0.2:
            death_prob *= 3
        
        return np.random.random() < death_prob
    
    async def _kill_cell(self, cell: Cell) -> Dict[str, Any]:
        """Remove cell from simulation"""
        
        # Remove from grid and cells dict
        self.grid[cell.position] = None
        del self.cells[cell.id]
        
        # Remove from neighbors' neighbor lists
        for neighbor_id in cell.neighbors:
            if neighbor_id in self.cells:
                self.cells[neighbor_id].neighbors.discard(cell.id)
        
        return {
            "type": "cell_death",
            "cell_id": cell.id,
            "cell_type": cell.cell_type.value,
            "age": cell.age,
            "step": self.simulation_step
        }
    
    def _sample_cell_type(self, distribution: Dict[CellType, float]) -> CellType:
        """Sample cell type from distribution"""
        
        types = list(distribution.keys())
        probabilities = list(distribution.values())
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        return np.random.choice(types, p=probabilities)
    
    def _find_empty_position(self) -> Optional[Tuple[int, int, int]]:
        """Find random empty position in grid"""
        
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.randint(0, self.grid_size[0])
            y = np.random.randint(0, self.grid_size[1])
            z = np.random.randint(0, self.grid_size[2])
            
            if self.grid[x, y, z] is None:
                return (x, y, z)
        
        return None  # Grid might be full
    
    def _find_adjacent_empty_position(self, position: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        """Find empty position adjacent to given position"""
        
        x, y, z = position
        
        # Check all 26 adjacent positions in 3D
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Check bounds
                    if (0 <= nx < self.grid_size[0] and 
                        0 <= ny < self.grid_size[1] and 
                        0 <= nz < self.grid_size[2]):
                        
                        if self.grid[nx, ny, nz] is None:
                            return (nx, ny, nz)
        
        return None
    
    async def _update_neighborhoods(self):
        """Update neighborhood relationships for all cells"""
        
        # Clear existing neighborhoods
        for cell in self.cells.values():
            cell.neighbors.clear()
        
        # Rebuild neighborhoods
        positions = {cell.id: cell.position for cell in self.cells.values()}
        
        for cell_id, position in positions.items():
            x, y, z = position
            
            # Check adjacent positions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        nx, ny, nz = x + dx, y + dy, z + dz
                        
                        # Check bounds
                        if (0 <= nx < self.grid_size[0] and 
                            0 <= ny < self.grid_size[1] and 
                            0 <= nz < self.grid_size[2]):
                            
                            neighbor_id = self.grid[nx, ny, nz]
                            if neighbor_id and neighbor_id != cell_id:
                                self.cells[cell_id].neighbors.add(neighbor_id)
    
    async def _update_molecular_diffusion(self):
        """Update molecular diffusion across tissue"""
        
        # Simplified diffusion - in practice would use proper PDE solvers
        
        # Create concentration fields
        for molecule in ["oxygen", "glucose", "growth_factors"]:
            # Get current concentrations
            concentrations = np.zeros(self.grid_size)
            
            for cell in self.cells.values():
                x, y, z = cell.position
                if molecule in cell.metabolites:
                    concentrations[x, y, z] = cell.metabolites[molecule]
                elif molecule in cell.signaling_molecules:
                    concentrations[x, y, z] = cell.signaling_molecules[molecule]
            
            # Apply diffusion (simple Gaussian blur)
            diffusion_coeff = self.diffusion_coefficients.get(molecule, 0.5)
            if diffusion_coeff > 0:
                concentrations = ndimage.gaussian_filter(concentrations, sigma=diffusion_coeff)
            
            # Update cell concentrations
            for cell in self.cells.values():
                x, y, z = cell.position
                new_concentration = concentrations[x, y, z]
                
                if molecule in cell.metabolites:
                    cell.metabolites[molecule] = new_concentration
                elif molecule in cell.signaling_molecules:
                    cell.signaling_molecules[molecule] = new_concentration
    
    async def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get current simulation statistics"""
        
        if not self.cells:
            return {"total_cells": 0}
        
        # Cell type distribution
        cell_type_counts = {}
        cell_state_counts = {}
        
        total_energy = 0
        total_health = 0
        total_age = 0
        
        for cell in self.cells.values():
            # Count by type
            cell_type_counts[cell.cell_type.value] = cell_type_counts.get(cell.cell_type.value, 0) + 1
            
            # Count by state
            cell_state_counts[cell.state.value] = cell_state_counts.get(cell.state.value, 0) + 1
            
            # Accumulate metrics
            total_energy += cell.energy
            total_health += cell.health
            total_age += cell.age
        
        num_cells = len(self.cells)
        
        return {
            "simulation_step": self.simulation_step,
            "total_cells": num_cells,
            "cell_type_distribution": cell_type_counts,
            "cell_state_distribution": cell_state_counts,
            "average_energy": total_energy / num_cells,
            "average_health": total_health / num_cells,
            "average_age": total_age / num_cells,
            "environment": {
                "oxygen_level": self.environment.oxygen_level,
                "nutrient_level": self.environment.nutrient_level,
                "toxin_level": self.environment.toxin_level,
                "inflammation_level": self.environment.inflammation_level
            }
        }
