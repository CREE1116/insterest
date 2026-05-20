import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Upload, Image as ImageIcon, Camera } from 'lucide-react';

interface ImageSearchModalProps {
  onClose: () => void;
  onSelectImage: (file: File) => void;
}

const ImageSearchModal: React.FC<ImageSearchModalProps> = ({ onClose, onSelectImage }) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onSelectImage(file);
        onClose();
      } else {
        alert('이미지 파일만 업로드할 수 있습니다.');
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onSelectImage(e.target.files[0]);
      onClose();
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      exit={{ opacity: 0 }} 
      className="modal-overlay" 
      onClick={onClose}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.6)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 50
      }}
    >
      <motion.div 
        initial={{ scale: 0.95, opacity: 0, y: 20 }} 
        animate={{ scale: 1, opacity: 1, y: 0 }} 
        exit={{ scale: 0.95, opacity: 0, y: 20 }}
        onClick={e => e.stopPropagation()} 
        style={{
          backgroundColor: 'white', 
          width: '90%', 
          maxWidth: '550px', 
          borderRadius: '32px', 
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.15)', 
          padding: '2.5rem',
          display: 'flex', 
          flexDirection: 'column',
          position: 'relative',
          border: '1px solid #f1f5f9'
        }}
      >
        {/* Close Button */}
        <button 
          onClick={onClose} 
          style={{ 
            position: 'absolute',
            top: '1.5rem',
            right: '1.5rem',
            background: '#f1f5f9',
            border: 'none',
            borderRadius: '50%',
            width: '2.5rem',
            height: '2.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            color: 'var(--text-main)',
            transition: 'background-color 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#e2e8f0'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#f1f5f9'}
        >
          <X size={20} />
        </button>

        {/* Title */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ 
            width: '3.5rem', 
            height: '3.5rem', 
            backgroundColor: '#fee2e2', 
            borderRadius: '50%', 
            display: 'inline-flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            marginBottom: '1rem',
            color: 'var(--primary-red)'
          }}>
            <Camera size={24} />
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 900, color: 'var(--text-main)', marginBottom: '0.5rem', letterSpacing: '-0.03em' }}>
            이미지 검색 및 추천
          </h2>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5', padding: '0 1rem' }}>
            찾고 싶은 분위기의 이미지를 드래그 앤 드롭하거나 클릭하여 선택하면, 유사한 콘텐츠와 유저 맞춤 피드를 찾아드립니다.
          </p>
        </div>

        {/* Drag and Drop Zone */}
        <div 
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={onButtonClick}
          style={{
            border: isDragActive ? '3px dashed var(--primary-red)' : '3px dashed #e2e8f0',
            borderRadius: '24px',
            padding: '4rem 2rem',
            textAlign: 'center',
            cursor: 'pointer',
            backgroundColor: isDragActive ? '#fff5f5' : '#fafafa',
            transition: 'all 0.2s ease-in-out',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem'
          }}
          onMouseOver={(e) => {
            if (!isDragActive) {
              e.currentTarget.style.borderColor = 'var(--primary-red)';
              e.currentTarget.style.backgroundColor = '#fff8f8';
            }
          }}
          onMouseOut={(e) => {
            if (!isDragActive) {
              e.currentTarget.style.borderColor = '#e2e8f0';
              e.currentTarget.style.backgroundColor = '#fafafa';
            }
          }}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleChange} 
            accept="image/*" 
            style={{ display: 'none' }} 
          />
          
          <motion.div
            animate={{ y: isDragActive ? -10 : 0 }}
            style={{
              width: '4rem',
              height: '4rem',
              borderRadius: '20px',
              backgroundColor: isDragActive ? '#fee2e2' : '#f1f5f9',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: isDragActive ? 'var(--primary-red)' : '#94a3b8',
              transition: 'background-color 0.2s'
            }}
          >
            {isDragActive ? <Upload size={28} /> : <ImageIcon size={28} />}
          </motion.div>

          <div>
            <p style={{ fontWeight: 800, color: 'var(--text-main)', fontSize: '1rem', marginBottom: '0.25rem' }}>
              {isDragActive ? '여기에 이미지를 놓으세요!' : '이미지를 드래그 앤 드롭하세요'}
            </p>
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', fontWeight: 500 }}>
              또는 클릭하여 기기에서 파일 선택
            </p>
          </div>
        </div>

        {/* Footer info */}
        <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          지원 형식: PNG, JPG, JPEG, WEBP (최대 10MB)
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ImageSearchModal;
