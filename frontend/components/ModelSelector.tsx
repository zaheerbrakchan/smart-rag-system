'use client';

import { useState, useRef, useEffect } from 'react';
import { ModelType } from '@/types';
import { ChevronDown, Sparkles, Cpu } from 'lucide-react';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onSelectModel: (model: ModelType) => void;
  disabled?: boolean;
}

const models = [
  {
    id: 'openai' as ModelType,
    name: 'GPT-4o-mini',
    provider: 'OpenAI',
    description: 'Fast and efficient GPT-4 variant',
    icon: Sparkles,
  },
  {
    id: 'huggingface' as ModelType,
    name: 'Mistral-7B-Instruct',
    provider: 'HuggingFace',
    description: 'Open-source instruction-tuned model',
    icon: Cpu,
  },
];

export default function ModelSelector({
  selectedModel,
  onSelectModel,
  disabled = false,
}: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selected = models.find((m) => m.id === selectedModel) || models[0];

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
        }`}
      >
        <selected.icon className="w-4 h-4 text-primary-600" />
        <span className="text-sm font-medium text-gray-700">{selected.name}</span>
        <ChevronDown
          className={`w-4 h-4 text-gray-500 transition-transform ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-72 bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden">
          <div className="p-2">
            <p className="text-xs text-gray-500 px-3 py-2 uppercase tracking-wider">
              Select Model
            </p>
            {models.map((model) => (
              <button
                key={model.id}
                onClick={() => {
                  onSelectModel(model.id);
                  setIsOpen(false);
                }}
                className={`w-full flex items-start gap-3 p-3 rounded-lg transition-colors ${
                  selectedModel === model.id
                    ? 'bg-primary-50 border border-primary-200'
                    : 'hover:bg-gray-50'
                }`}
              >
                <model.icon
                  className={`w-5 h-5 mt-0.5 ${
                    selectedModel === model.id ? 'text-primary-600' : 'text-gray-400'
                  }`}
                />
                <div className="text-left">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-medium ${
                        selectedModel === model.id ? 'text-primary-700' : 'text-gray-700'
                      }`}
                    >
                      {model.name}
                    </span>
                    <span className="text-xs text-gray-400">{model.provider}</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">{model.description}</p>
                </div>
                {selectedModel === model.id && (
                  <div className="ml-auto">
                    <div className="w-2 h-2 bg-primary-500 rounded-full" />
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
