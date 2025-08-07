import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface SequenceEditorProps {
  sequence: string;
  analysisResult?: any;
  readOnly?: boolean;
  onChange?: (sequence: string) => void;
}

export const SequenceEditor: React.FC<SequenceEditorProps> = ({ 
  sequence, 
  analysisResult, 
  readOnly = false,
  onChange 
}) => {
  const [displaySequence, setDisplaySequence] = useState(sequence);
  const [selectedRange, setSelectedRange] = useState<[number, number] | null>(null);

  useEffect(() => {
    setDisplaySequence(sequence);
  }, [sequence]);

  const handleSequenceChange = (newSequence: string) => {
    if (!readOnly) {
      setDisplaySequence(newSequence);
      onChange?.(newSequence);
    }
  };

  const getAminoAcidColor = (aa: string, position: number) => {
    // Color amino acids by properties
    const hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'];
    const polar = ['S', 'T', 'N', 'Q'];
    const charged = ['D', 'E', 'K', 'R'];
    const special = ['C', 'G', 'P', 'H'];

    if (hydrophobic.includes(aa)) return 'bg-blue-100 text-blue-800';
    if (polar.includes(aa)) return 'bg-green-100 text-green-800';
    if (charged.includes(aa)) return 'bg-red-100 text-red-800';
    if (special.includes(aa)) return 'bg-yellow-100 text-yellow-800';
    return 'bg-gray-100 text-gray-800';
  };

  const formatSequenceWithHighlights = () => {
    const chars = displaySequence.split('');
    const chunkSize = 10;
    const chunks = [];

    for (let i = 0; i < chars.length; i += chunkSize) {
      chunks.push(chars.slice(i, i + chunkSize));
    }

    return chunks.map((chunk, chunkIndex) => (
      <div key={chunkIndex} className="flex items-center mb-2">
        <div className="w-12 text-xs text-gray-500 mr-4">
          {chunkIndex * chunkSize + 1}
        </div>
        <div className="flex space-x-1">
          {chunk.map((aa, aaIndex) => {
            const globalIndex = chunkIndex * chunkSize + aaIndex;
            const isHighlighted = analysisResult?.functional_domains?.some((domain: any) => 
              globalIndex >= domain.start - 1 && globalIndex <= domain.end - 1
            );
            
            return (
              <motion.span
                key={globalIndex}
                className={`
                  px-1 py-0.5 text-xs font-mono rounded cursor-pointer
                  ${getAminoAcidColor(aa, globalIndex)}
                  ${isHighlighted ? 'ring-2 ring-purple-400' : ''}
                `}
                whileHover={{ scale: 1.1 }}
                onClick={() => setSelectedRange([globalIndex, globalIndex])}
                title={`${aa} at position ${globalIndex + 1}`}
              >
                {aa}
              </motion.span>
            );
          })}
        </div>
      </div>
    ));
  };

  return (
    <div className="space-y-4">
      {/* Sequence Display */}
      <div className="bg-gray-50 p-4 rounded-lg border">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-gray-900">Amino Acid Sequence</h4>
          <div className="text-xs text-gray-500">
            Length: {displaySequence.length} residues
          </div>
        </div>
        
        <div className="font-mono text-sm max-h-64 overflow-y-auto">
          {formatSequenceWithHighlights()}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-100 rounded mr-1"></div>
          <span>Hydrophobic</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-100 rounded mr-1"></div>
          <span>Polar</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-100 rounded mr-1"></div>
          <span>Charged</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-yellow-100 rounded mr-1"></div>
          <span>Special</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 border-2 border-purple-400 rounded mr-1"></div>
          <span>Functional Domain</span>
        </div>
      </div>

      {/* Functional Domains */}
      {analysisResult?.functional_domains && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-900">Functional Domains</h4>
          {analysisResult.functional_domains.map((domain: any, index: number) => (
            <div key={index} className="flex items-center justify-between p-2 bg-purple-50 rounded border border-purple-200">
              <div>
                <span className="text-sm font-medium text-purple-900">{domain.name}</span>
                <span className="text-xs text-purple-600 ml-2">
                  {domain.start}-{domain.end}
                </span>
              </div>
              <div className="text-xs text-purple-600">
                {(domain.confidence * 100).toFixed(0)}% confidence
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
