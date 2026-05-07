import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Wand2, Upload, Music, Image as ImageIcon, Play, Pause, X, ChevronRight, AlertCircle, FolderPlus } from 'lucide-react';
import client from '../api/client';

interface CreateModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const CreateModal: React.FC<CreateModalProps> = ({ onClose, onSuccess }) => {
  const [mode, setMode] = useState<'ai' | 'upload'>('ai');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [caption, setCaption] = useState('');
  const [hashtagInput, setHashtagInput] = useState('');
  
  const [prompt, setPrompt] = useState('');
  const [genResult, setGenResult] = useState<{ image_url: string, audio_url: string } | null>(null);

  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(10);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (selectedAudio) {
      const url = URL.createObjectURL(selectedAudio);
      setAudioPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setAudioPreviewUrl(null);
      setIsPlaying(false);
    }
  }, [selectedAudio]);

  useEffect(() => {
    if (selectedImage) {
      const url = URL.createObjectURL(selectedImage);
      setImagePreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setImagePreviewUrl(null);
      setSelectedAudio(null);
    }
  }, [selectedImage]);

  const handleStartAI = () => {
    if (!prompt.trim()) return;
    setIsGenerating(true);
    setProgress(10);
    setStatusText('AI 아티스트를 불러오는 중...');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/ws/generate`);

    ws.onopen = () => ws.send(JSON.stringify({ user_input: prompt }));
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.message) setStatusText(msg.message);
      if (msg.progress) setProgress(msg.progress);
      if (msg.status === 'completed') {
        setGenResult(msg.data);
        setIsGenerating(false);
        ws.close();
      }
    };
    ws.onerror = () => { alert('생성 서버 연결 실패'); setIsGenerating(false); };
  };

  const handleAudioLoaded = (e: React.SyntheticEvent<HTMLAudioElement>) => {
    const dur = e.currentTarget.duration;
    setDuration(dur);
    setEndTime(Math.min(dur, 10));
  };

  const handleStartTimeChange = (newStart: number) => {
    setStartTime(newStart);
    setEndTime(Math.min(duration, newStart + 10));
    if (audioRef.current) {
      audioRef.current.currentTime = newStart;
    }
  };

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.currentTime = startTime;
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleFinish = async () => {
    try {
      const hashtags = hashtagInput.split(',').map((tag: string) => tag.trim().replace(/^#/, '')).filter((tag: string) => tag);
      
      if (mode === 'ai' && genResult) {
        await client.post('/upload/content/', {
          metadata_info: {
            image_url: genResult.image_url,
            audio_url: genResult.audio_url,
            prompt: prompt
          },
          caption: caption || prompt,
          hashtags: ['AI', ...hashtags]
        });
      } else if (selectedImage) {
        const formData = new FormData();
        formData.append('image_file', selectedImage);
        if (selectedAudio) {
          formData.append('audio_file', selectedAudio);
          formData.append('start_time', startTime.toString());
          formData.append('end_time', endTime.toString());
        }

        const res = await client.post('/upload/media/upload-combined', formData);
        await client.post('/upload/content/', {
          content_id: res.data.content_id,
          caption: caption,
          hashtags: hashtags
        });
      }

      onSuccess();
      onClose();
    } catch (e) { alert('게시 실패: ' + e); }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="modal-overlay" onClick={onClose}>
      <motion.div 
        initial={{ scale: 0.95, opacity: 0, y: 20 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.95, opacity: 0, y: 20 }}
        onClick={e => e.stopPropagation()} 
        className="create-modal-content"
        style={{
          backgroundColor: 'white', width: '90%', maxWidth: '1000px', borderRadius: '40px', maxHeight: '90vh', overflow: 'hidden', boxShadow: 'var(--shadow-lg)', display: 'flex', flexDirection: 'column'
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2rem 3rem', borderBottom: '1px solid #f1f5f9' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 900, letterSpacing: '-0.04em' }}>새로운 창작</h2>
          <div style={{ display: 'flex', gap: '0.5rem', backgroundColor: '#f1f5f9', padding: '0.4rem', borderRadius: 'var(--radius-full)' }}>
            <button onClick={() => setMode('ai')} style={{ padding: '0.5rem 1.25rem', borderRadius: 'var(--radius-full)', fontWeight: 800, fontSize: '0.8125rem', backgroundColor: mode === 'ai' ? 'white' : 'transparent', boxShadow: mode === 'ai' ? 'var(--shadow-sm)' : 'none' }}>AI 생성</button>
            <button onClick={() => setMode('upload')} style={{ padding: '0.5rem 1.25rem', borderRadius: 'var(--radius-full)', fontWeight: 800, fontSize: '0.8125rem', backgroundColor: mode === 'upload' ? 'white' : 'transparent', boxShadow: mode === 'upload' ? 'var(--shadow-sm)' : 'none' }}>직접 업로드</button>
          </div>
          <button onClick={onClose} style={{ padding: '0.5rem' }}><X size={24} /></button>
        </div>

        <div className="create-modal-split" style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Left: Preview Section */}
          <div style={{ flex: 1.2, backgroundColor: '#f8fafc', padding: '3rem', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRight: '1px solid #f1f5f9' }}>
            <div style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <AnimatePresence mode="wait">
                {mode === 'ai' ? (
                   <motion.div key="ai-preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      {genResult ? (
                        <div style={{ position: 'relative', width: '100%', borderRadius: '24px', overflow: 'hidden', boxShadow: 'var(--shadow-md)' }}>
                          <img src={genResult.image_url} style={{ width: '100%', display: 'block' }} alt="" />
                          <div style={{ position: 'absolute', bottom: '1.5rem', left: '1.5rem', right: '1.5rem' }}>
                            <audio src={genResult.audio_url} controls style={{ width: '100%', height: '40px' }} />
                          </div>
                        </div>
                      ) : (
                        <div style={{ width: '100%', height: '100%', backgroundColor: '#e2e8f0', borderRadius: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem', color: '#94a3b8' }}>
                          <Wand2 size={48} />
                          <span style={{ fontWeight: 700 }}>AI가 당신의 상상을 그립니다</span>
                        </div>
                      )}
                   </motion.div>
                ) : (
                  <motion.div key="upload-preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ width: '100%', height: '100%', position: 'relative' }}>
                    {imagePreviewUrl ? (
                      <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                        <img src={imagePreviewUrl} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '24px' }} alt="" />
                        <button onClick={() => setSelectedImage(null)} style={{ position: 'absolute', top: '1rem', right: '1rem', backgroundColor: 'rgba(0,0,0,0.5)', color: 'white', padding: '0.5rem', borderRadius: '50%' }}><X size={16} /></button>
                        {audioPreviewUrl && (
                          <div style={{ position: 'absolute', bottom: '1rem', left: '1rem', right: '1rem', backgroundColor: 'rgba(255,255,255,0.9)', padding: '1rem', borderRadius: '16px', backdropFilter: 'blur(10px)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <button onClick={togglePlay} style={{ width: '2.5rem', height: '2.5rem', backgroundColor: 'var(--primary-red)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '50%' }}>
                              {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" style={{ marginLeft: '2px' }} />}
                            </button>
                            <div style={{ flex: 1 }}>
                              <div style={{ height: '4px', backgroundColor: '#e2e8f0', borderRadius: '2px', position: 'relative' }}>
                                <div style={{ position: 'absolute', left: 0, width: '40%', height: '100%', backgroundColor: 'var(--primary-red)', borderRadius: '2px' }} />
                              </div>
                              <span style={{ fontSize: '0.75rem', fontWeight: 700, marginTop: '4px', display: 'block' }}>배경음악 재생 중...</span>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <label style={{ width: '100%', height: '100%', border: '3px dashed #e2e8f0', borderRadius: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', gap: '1rem', backgroundColor: 'white' }}>
                        <div style={{ width: '4rem', height: '4rem', backgroundColor: '#fee2e2', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <ImageIcon size={32} color="var(--primary-red)" />
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <p style={{ fontWeight: 800, color: 'var(--text-main)' }}>클릭하여 이미지 업로드</p>
                          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>PNG, JPG, WEBP 지원</p>
                        </div>
                        <input type="file" accept="image/*" hidden onChange={e => setSelectedImage(e.target.files?.[0] || null)} />
                      </label>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Right: Controls Section */}
          <div style={{ flex: 1, padding: '3rem', overflowY: 'auto' }}>
            <AnimatePresence mode="wait">
              {mode === 'ai' ? (
                <motion.div key="ai-ctrl" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                  {!isGenerating && !genResult ? (
                    <>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <label style={{ fontWeight: 800, fontSize: '0.875rem', color: 'var(--text-muted)' }}>프롬프트 입력</label>
                        <textarea 
                          placeholder="어떤 콘텐츠를 만들고 싶나요?"
                          value={prompt} onChange={e => setPrompt(e.target.value)}
                          style={{ width: '100%', height: '150px', padding: '1.5rem', borderRadius: '20px', backgroundColor: '#f1f5f9', border: 'none', outline: 'none', resize: 'none', fontSize: '1rem', fontWeight: 600 }}
                        />
                      </div>
                      <button onClick={handleStartAI} disabled={!prompt.trim()} style={{ backgroundColor: 'var(--primary-red)', color: 'white', padding: '1.25rem', borderRadius: 'var(--radius-full)', fontWeight: 900, fontSize: '1.125rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', opacity: prompt.trim() ? 1 : 0.5 }}>
                        <Wand2 size={20} /> AI 아트 생성하기
                      </button>
                    </>
                  ) : isGenerating ? (
                    <div style={{ textAlign: 'center', padding: '2rem 0' }}>
                      <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }} style={{ width: '4rem', height: '4rem', border: '4px solid #f1f5f9', borderTopColor: 'var(--primary-red)', borderRadius: '50%', margin: '0 auto 2rem' }} />
                      <h3 style={{ fontWeight: 900, fontSize: '1.25rem', marginBottom: '0.5rem' }}>{statusText}</h3>
                      <p style={{ color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.875rem' }}>잠시만 기다려주세요</p>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <label style={{ fontWeight: 800, fontSize: '0.875rem', color: 'var(--text-muted)' }}>게시물 설명</label>
                        <input placeholder="이 작품에 대해 설명해주세요..." value={caption} onChange={e => setCaption(e.target.value)} style={{ padding: '1.25rem', borderRadius: '16px', border: '2px solid #f1f5f9', backgroundColor: 'white', outline: 'none', fontWeight: 600 }} />
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <label style={{ fontWeight: 800, fontSize: '0.875rem', color: 'var(--text-muted)' }}>해시태그 (선택)</label>
                        <input placeholder="쉼표(,)로 구분하여 입력 (예: 일상, 감성)" value={hashtagInput} onChange={e => setHashtagInput(e.target.value)} style={{ padding: '1.25rem', borderRadius: '16px', border: '2px solid #f1f5f9', backgroundColor: 'white', outline: 'none', fontWeight: 600 }} />
                      </div>
                      <button onClick={handleFinish} style={{ backgroundColor: 'var(--black)', color: 'white', padding: '1.25rem', borderRadius: 'var(--radius-full)', fontWeight: 900, fontSize: '1.125rem' }}>피드에 공유하기</button>
                      <button onClick={() => { setGenResult(null); setPrompt(''); }} style={{ color: 'var(--text-muted)', fontWeight: 700 }}>다시 만들기</button>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div key="upload-ctrl" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <label style={{ fontWeight: 800, fontSize: '0.875rem', color: 'var(--text-muted)' }}>설명 입력</label>
                    <input placeholder="무슨 생각을 하고 계신가요?" value={caption} onChange={e => setCaption(e.target.value)} style={{ padding: '1.25rem', borderRadius: '16px', border: '2px solid #f1f5f9', backgroundColor: 'white', outline: 'none', fontWeight: 600 }} />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <label style={{ fontWeight: 800, fontSize: '0.875rem', color: 'var(--text-muted)' }}>해시태그 (선택)</label>
                    <input placeholder="쉼표(,)로 구분 (예: OOTD, 고양이)" value={hashtagInput} onChange={e => setHashtagInput(e.target.value)} style={{ padding: '1.25rem', borderRadius: '16px', border: '2px solid #f1f5f9', backgroundColor: 'white', outline: 'none', fontWeight: 600 }} />
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label style={{ fontWeight: 800, fontSize: '0.875rem', color: 'var(--text-muted)' }}>배경 음악 (선택)</label>
                      {selectedAudio && <button onClick={() => setSelectedAudio(null)} style={{ color: 'var(--primary-red)', fontSize: '0.75rem', fontWeight: 800 }}>취소</button>}
                    </div>
                    
                    {!selectedImage ? (
                       <div style={{ padding: '1.5rem', backgroundColor: '#f8fafc', borderRadius: '20px', display: 'flex', alignItems: 'center', gap: '1rem', border: '1px solid #f1f5f9', opacity: 0.6 }}>
                          <AlertCircle size={20} color="#94a3b8" />
                          <span style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#94a3b8' }}>사진을 먼저 업로드해주세요.</span>
                       </div>
                    ) : !selectedAudio ? (
                      <label style={{ padding: '1.5rem', backgroundColor: 'white', border: '2px solid #f1f5f9', borderRadius: '20px', display: 'flex', alignItems: 'center', gap: '1rem', cursor: 'pointer' }}>
                        <div style={{ width: '2.5rem', height: '2.5rem', backgroundColor: '#f1f5f9', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Music size={18} color="var(--text-main)" />
                        </div>
                        <span style={{ fontWeight: 700, fontSize: '0.875rem' }}>음악 파일 선택하기</span>
                        <input type="file" accept="audio/*" hidden onChange={e => setSelectedAudio(e.target.files?.[0] || null)} />
                      </label>
                    ) : (
                      <div style={{ padding: '2rem', backgroundColor: '#f8fafc', borderRadius: '24px', border: '1px solid #f1f5f9' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem', alignItems: 'center' }}>
                          <span style={{ fontWeight: 800, fontSize: '0.875rem' }}>구간 선택 (최대 10초)</span>
                          <span style={{ fontSize: '0.75rem', backgroundColor: 'white', padding: '0.25rem 0.6rem', borderRadius: 'var(--radius-full)', fontWeight: 800, color: 'var(--primary-red)' }}>
                            {startTime.toFixed(1)}s ~ {endTime.toFixed(1)}s
                          </span>
                        </div>
                        
                        <div style={{ position: 'relative', height: '8px', backgroundColor: '#e2e8f0', borderRadius: '4px', marginBottom: '2.5rem' }}>
                          <div style={{ position: 'absolute', left: `${(startTime / (duration || 1)) * 100}%`, width: `${((endTime - startTime) / (duration || 1)) * 100}%`, height: '100%', backgroundColor: 'var(--primary-red)', borderRadius: '4px' }} />
                          <input 
                            type="range" min={0} max={Math.max(0, duration - 1)} step={0.1} value={startTime} 
                            onChange={e => handleStartTimeChange(parseFloat(e.target.value))}
                            style={{ position: 'absolute', top: '-10px', left: 0, width: '100%', opacity: 0, cursor: 'pointer' }} 
                          />
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'center' }}>
                          <button onClick={togglePlay} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.25rem', backgroundColor: 'white', borderRadius: 'var(--radius-full)', boxShadow: 'var(--shadow-sm)', fontWeight: 800, fontSize: '0.8125rem' }}>
                            {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
                            미리듣기
                          </button>
                        </div>
                        <audio ref={audioRef} src={audioPreviewUrl || ''} onLoadedMetadata={handleAudioLoaded} onEnded={() => setIsPlaying(false)} hidden />
                      </div>
                    )}
                  </div>

                  <button 
                    onClick={handleFinish} 
                    disabled={!selectedImage}
                    style={{ 
                      marginTop: 'auto', backgroundColor: 'var(--primary-red)', color: 'white', padding: '1.25rem', borderRadius: 'var(--radius-full)', fontWeight: 900, fontSize: '1.125rem', boxShadow: '0 10px 20px -5px rgba(230, 0, 35, 0.3)',
                      opacity: selectedImage ? 1 : 0.5, transition: 'all 0.3s'
                    }}
                  >
                    공유하기
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default CreateModal;
