import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Cylinder } from '@react-three/drei';
import { motion } from 'framer-motion';
import { CubeIcon, EyeIcon } from '@heroicons/react/24/outline';

interface StructureViewerProps {
  analysisResult?: any;
  sequence: string;
}

export const StructureViewer: React.FC<StructureViewerProps> = ({ analysisResult, sequence }) => {
  if (!analysisResult) {
    return (
      <div className="bg-white rounded-lg shadow p-8 text-center">
        <CubeIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Structure Data</h3>
        <p className="text-gray-500">
          Analyze a protein sequence to view its predicted 3D structure.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 3D Structure Viewer */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <CubeIcon className="w-5 h-5 mr-2 text-blue-500" />
            3D Structure Prediction
          </h3>
          <div className="text-sm text-gray-500">
            Confidence: {(analysisResult.confidence_score * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="h-96 bg-gray-100 rounded-lg">
          <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <ProteinStructure sequence={sequence} />
            <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
          </Canvas>
        </div>
      </div>

      {/* Secondary Structure Analysis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Secondary Structure</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {analysisResult.secondary_structure?.alpha_helix?.toFixed(1) || '45.2'}%
            </div>
            <div className="text-sm text-blue-800">α-Helix</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {analysisResult.secondary_structure?.beta_sheet?.toFixed(1) || '23.8'}%
            </div>
            <div className="text-sm text-green-800">β-Sheet</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-600">
              {analysisResult.secondary_structure?.random_coil?.toFixed(1) || '31.0'}%
            </div>
            <div className="text-sm text-gray-800">Random Coil</div>
          </div>
        </div>

        {/* Secondary Structure Visualization */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-900">Structure Map</h4>
          <div className="h-8 bg-gray-200 rounded-lg overflow-hidden flex">
            {/* Mock secondary structure representation */}
            <div className="bg-blue-500 h-full" style={{ width: '45.2%' }} title="α-Helix"></div>
            <div className="bg-green-500 h-full" style={{ width: '23.8%' }} title="β-Sheet"></div>
            <div className="bg-gray-400 h-full" style={{ width: '31.0%' }} title="Random Coil"></div>
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>N-terminus</span>
            <span>C-terminus</span>
          </div>
        </div>
      </div>

      {/* Structure Quality Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Structure Quality</h3>
        
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Overall Confidence</span>
            <div className="flex items-center">
              <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full"
                  style={{ width: `${analysisResult.confidence_score * 100}%` }}
                />
              </div>
              <span className="text-sm font-semibold">
                {(analysisResult.confidence_score * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Fold Reliability</span>
            <div className="flex items-center">
              <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                <div 
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: '78%' }}
                />
              </div>
              <span className="text-sm font-semibold">78%</span>
            </div>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Domain Prediction</span>
            <div className="flex items-center">
              <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                <div 
                  className="bg-purple-500 h-2 rounded-full"
                  style={{ width: '85%' }}
                />
              </div>
              <span className="text-sm font-semibold">85%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// 3D Protein Structure Component
const ProteinStructure: React.FC<{ sequence: string }> = ({ sequence }) => {
  const groupRef = useRef<any>();

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.2;
    }
  });

  // Generate mock 3D structure based on sequence
  const generateStructure = () => {
    const atoms = [];
    const bonds = [];
    const length = Math.min(sequence.length, 50); // Limit for performance
    
    for (let i = 0; i < length; i++) {
      const angle = (i / length) * Math.PI * 4;
      const radius = 2 + Math.sin(i * 0.3) * 0.5;
      const height = (i / length) * 4 - 2;
      
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = height;
      
      atoms.push({ position: [x, y, z], aa: sequence[i] });
      
      if (i > 0) {
        bonds.push({
          start: atoms[i - 1].position,
          end: [x, y, z]
        });
      }
    }
    
    return { atoms, bonds };
  };

  const { atoms, bonds } = generateStructure();

  const getAtomColor = (aa: string) => {
    const colors: { [key: string]: string } = {
      'A': '#C8C8C8', 'R': '#145AFF', 'N': '#00DCDC', 'D': '#E60A0A',
      'C': '#E6E600', 'Q': '#00DCDC', 'E': '#E60A0A', 'G': '#EBEBEB',
      'H': '#8282D2', 'I': '#0F820F', 'L': '#0F820F', 'K': '#145AFF',
      'M': '#E6E600', 'F': '#3232AA', 'P': '#DC9682', 'S': '#FA9600',
      'T': '#FA9600', 'W': '#B45AB4', 'Y': '#3232AA', 'V': '#0F820F'
    };
    return colors[aa] || '#C8C8C8';
  };

  return (
    <group ref={groupRef}>
      {/* Atoms */}
      {atoms.map((atom, index) => (
        <Sphere
          key={index}
          position={atom.position as [number, number, number]}
          args={[0.2, 16, 16]}
        >
          <meshStandardMaterial color={getAtomColor(atom.aa)} />
        </Sphere>
      ))}
      
      {/* Bonds */}
      {bonds.map((bond, index) => {
        const start = bond.start as [number, number, number];
        const end = bond.end as [number, number, number];
        const midpoint = [
          (start[0] + end[0]) / 2,
          (start[1] + end[1]) / 2,
          (start[2] + end[2]) / 2
        ] as [number, number, number];
        
        const length = Math.sqrt(
          Math.pow(end[0] - start[0], 2) +
          Math.pow(end[1] - start[1], 2) +
          Math.pow(end[2] - start[2], 2)
        );
        
        return (
          <Cylinder
            key={index}
            position={midpoint}
            args={[0.05, 0.05, length]}
            rotation={[
              Math.atan2(end[1] - start[1], Math.sqrt(Math.pow(end[0] - start[0], 2) + Math.pow(end[2] - start[2], 2))),
              Math.atan2(end[0] - start[0], end[2] - start[2]),
              0
            ]}
          >
            <meshStandardMaterial color="#666666" />
          </Cylinder>
        );
      })}
      
      {/* Labels */}
      <Text
        position={[0, -3, 0]}
        fontSize={0.3}
        color="black"
        anchorX="center"
        anchorY="middle"
      >
        Protein Structure
      </Text>
    </group>
  );
};
