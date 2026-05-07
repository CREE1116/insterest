import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { X, Check } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import client from '../api/client';

interface SettingsModalProps {
  onClose: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ onClose }) => {
  const { user, checkAuth } = useAuth();
  const [nickname, setNickname] = useState(user?.nickname || '');
  const [bio, setBio] = useState(user?.bio || '');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState('');

  const handleUpdateProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage('');
    try {
      await client.put('/users/me', { nickname, bio });
      await checkAuth();
      setMessage('프로필이 업데이트되었습니다.');
      setTimeout(onClose, 1500);
    } catch (err) {
      setMessage('업데이트 실패');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="modal-overlay" onClick={onClose}>
      <motion.div 
        initial={{ scale: 0.9, opacity: 0, y: 20 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.9, opacity: 0, y: 20 }}
        onClick={e => e.stopPropagation()} 
        style={{ backgroundColor: 'white', width: '100%', maxWidth: '450px', padding: '3rem', borderRadius: '32px', boxShadow: 'var(--shadow-xl)' }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem' }}>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 900, letterSpacing: '-0.02em' }}>프로필 설정</h2>
          <button onClick={onClose} style={{ padding: '0.625rem', borderRadius: '50%', backgroundColor: '#f1f5f9', color: 'var(--text-main)', border: 'none', cursor: 'pointer' }}><X size={20} /></button>
        </div>

        <form onSubmit={handleUpdateProfile} style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
              <label style={{ fontSize: '0.8125rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>로그인 계정</label>
              <input value={user?.email} disabled style={{ backgroundColor: '#f8fafc', color: '#94a3b8', cursor: 'not-allowed', padding: '1rem 1.25rem', borderRadius: '16px', border: '1px solid #e2e8f0', fontSize: '0.9375rem', fontWeight: 600 }} />
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
              <label style={{ fontSize: '0.8125rem', fontWeight: 800, color: 'var(--text-main)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>닉네임</label>
              <input 
                placeholder="사용할 닉네임을 입력하세요" 
                value={nickname} 
                onChange={e => setNickname(e.target.value)} 
                style={{ backgroundColor: '#f1f5f9', padding: '1rem 1.25rem', borderRadius: '16px', border: 'none', fontSize: '1rem', fontWeight: 700, outline: 'none' }} 
              />
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
              <label style={{ fontSize: '0.8125rem', fontWeight: 800, color: 'var(--text-main)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>소개 (Bio)</label>
              <textarea 
                placeholder="당신의 무드를 한 줄로 표현해보세요" 
                value={bio} 
                onChange={e => setBio(e.target.value)} 
                style={{ backgroundColor: '#f1f5f9', borderRadius: '20px', padding: '1.25rem', border: 'none', resize: 'none', height: '120px', fontSize: '1rem', fontWeight: 600, lineHeight: '1.5', outline: 'none' }} 
              />
            </div>
          </div>

          {message && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} style={{ padding: '1.25rem', borderRadius: '16px', backgroundColor: '#ecfdf5', color: '#059669', fontSize: '0.9375rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Check size={18} /> {message}
            </motion.div>
          )}

          <button 
            type="submit" 
            disabled={isSubmitting || (nickname === user?.nickname && bio === user?.bio)} 
            style={{ 
              backgroundColor: 'var(--black)', 
              color: 'white', 
              padding: '1.25rem', 
              borderRadius: '999px', 
              fontWeight: 900, 
              fontSize: '1.0625rem',
              transition: 'all 0.2s',
              opacity: (isSubmitting || (nickname === user?.nickname && bio === user?.bio)) ? 0.5 : 1,
              cursor: (isSubmitting || (nickname === user?.nickname && bio === user?.bio)) ? 'not-allowed' : 'pointer',
              border: 'none'
            }}
          >
            {isSubmitting ? '저장 중...' : '저장하기'}
          </button>
        </form>
      </motion.div>
    </motion.div>
  );
};

export default SettingsModal;
