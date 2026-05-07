import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import client from '../api/client';

interface AuthModalProps {
  onClose: () => void;
}

const AuthModal: React.FC<AuthModalProps> = ({ onClose }) => {
  const { login } = useAuth();
  const [isLoginTab, setIsLoginTab] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [nickname, setNickname] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (isLoginTab) {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);
        const res = await client.post('/auth/login', formData);
        await login(res.data.access_token);
        onClose();
      } else {
        await client.post('/auth/register', { email, password, nickname: nickname });
        alert('가입 완료! 로그인 해주세요.');
        setIsLoginTab(true);
      }
    } catch (err: any) {
      alert(err.response?.data?.detail || '요청 실패');
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="modal-overlay" 
      onClick={onClose}
    >
      <motion.div 
        initial={{ scale: 0.9, y: 40, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        exit={{ scale: 0.9, y: 40, opacity: 0 }}
        onClick={e => e.stopPropagation()} 
        style={{
          backgroundColor: 'white',
          width: '100%',
          maxWidth: '400px',
          padding: '40px',
          borderRadius: '32px',
          textAlign: 'center',
          boxShadow: 'var(--shadow-lg)'
        }}
      >
        <h1 style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--primary-red)', marginBottom: '8px', letterSpacing: '-0.05em' }}>Interest</h1>
        <p style={{ color: '#111', fontSize: '1.25rem', fontWeight: 700, marginBottom: '32px' }}>
          {isLoginTab ? '로그인하여 더 많은 아이디어를 만나보세요' : '가입하고 영감을 받아보세요'}
        </p>
        
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {!isLoginTab && (
            <input 
              type="text" placeholder="닉네임" required
              value={nickname} onChange={e => setNickname(e.target.value)}
              style={{ padding: '14px 20px', borderRadius: '16px', fontSize: '1rem', border: '2px solid #efefef' }}
            />
          )}

          <input 
            type="email" placeholder="이메일" required
            value={email} onChange={e => setEmail(e.target.value)}
            style={{ padding: '14px 20px', borderRadius: '16px', fontSize: '1rem', border: '2px solid #efefef' }}
          />
          <input 
            type="password" placeholder="비밀번호" required
            value={password} onChange={e => setPassword(e.target.value)}
            style={{ padding: '14px 20px', borderRadius: '16px', fontSize: '1rem', border: '2px solid #efefef' }}
          />
          
          <button type="submit" disabled={loading} style={{
            backgroundColor: 'var(--primary-red)',
            color: 'white',
            padding: '14px',
            borderRadius: '999px',
            fontWeight: 700,
            fontSize: '1rem',
            marginTop: '8px',
            opacity: loading ? 0.7 : 1
          }}>
            {loading ? '대기 중...' : (isLoginTab ? '로그인' : '계속하기')}
          </button>
        </form>

        <div style={{ marginTop: '24px' }}>
          <p style={{ fontSize: '0.875rem', fontWeight: 700, color: '#111' }}>
            {isLoginTab ? '아직 회원이 아니신가요?' : '이미 회원이신가요?'}
            <button 
              onClick={() => setIsLoginTab(!isLoginTab)}
              style={{ marginLeft: '8px', color: '#0070f3', fontWeight: 700 }}
            >
              {isLoginTab ? '회원가입' : '로그인'}
            </button>
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default AuthModal;
