import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Volume2, VolumeX, Heart, MessageCircle, Bookmark, Trash2, Send, FolderPlus, ChevronRight, Edit3 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import client from '../api/client';
import { usePostStats, usePostComments, useCollections, useToggleLikeMutation, useCommentMutation } from '../hooks/usePostQueries';
import { useQueryClient } from '@tanstack/react-query';

interface PostModalProps {
  post: any;
  isSavedInitial?: boolean;
  onClose: () => void;
  onDelete?: () => void;
}

const PostModal: React.FC<PostModalProps> = ({ post, isSavedInitial = false, onClose, onDelete }) => {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const audioRef = useRef<HTMLAudioElement>(null);
  
  // React Query Hooks (Declarative Data Fetching)
  const { data: stats } = usePostStats(post.id, { 
    likes: post.like_count ?? 0, 
    is_liked: !!post.is_liked 
  });
  const { data: comments } = usePostComments(post.id);
  
  // Mutations
  const toggleLike = useToggleLikeMutation();
  const addComment = useCommentMutation();
  
  // Local UI States
  const [showSaveMenu, setShowSaveMenu] = useState(false);
  const { data: collections } = useCollections(showSaveMenu);
  
  const [newComment, setNewComment] = useState('');
  const [isMuted, setIsMuted] = useState(true);
  const [progress, setProgress] = useState(0);
  const [isSaved, setIsSaved] = useState(isSavedInitial);
  const [isEditing, setIsEditing] = useState(false);
  const [editCaption, setEditCaption] = useState(post.caption);
  const [editHashtags, setEditHashtags] = useState(post.hashtags?.map((h: any) => h.tag).join(', ') || '');
  const [newCollectionName, setNewCollectionName] = useState('');
  const [saveStatus, setSaveStatus] = useState('');
  const saveMenuRef = useRef<HTMLDivElement>(null);

  // 외부 클릭 시 저장 메뉴 닫기
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (saveMenuRef.current && !saveMenuRef.current.contains(event.target as Node)) {
        setShowSaveMenu(false);
      }
    };
    if (showSaveMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showSaveMenu]);

  // Audio Progress Tracker
  useEffect(() => {
    const interval = setInterval(() => {
      if (audioRef.current) {
        const current = audioRef.current.currentTime;
        const duration = audioRef.current.duration;
        if (duration) setProgress((current / duration) * 100);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [post.id]);

  // Event Handlers
  const handleToggleLike = () => {
    if (!user) return alert('로그인이 필요합니다.');
    toggleLike.mutate(post.id);
  };

  const handleAddComment = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newComment.trim()) return;
    addComment.mutate({ postId: post.id, content: newComment }, {
      onSuccess: () => setNewComment(''),
      onError: () => alert('댓글 작성 실패')
    });
  };

  const handleDeletePost = async () => {
    if (!window.confirm('이 게시물을 정말 삭제하시겠습니까?')) return;
    try {
      await client.delete(`/upload/content/${post.id}`);
      alert('게시물이 삭제되었습니다.');
      if (onDelete) onDelete();
      onClose();
    } catch (e) {
      alert('게시물 삭제 실패');
    }
  };

  const handleUpdatePost = async () => {
    try {
      const hashtags = editHashtags.split(',').map((tag: string) => tag.trim().replace(/^#/, '')).filter((tag: string) => tag);
      await client.patch(`/upload/content/${post.id}`, {
        caption: editCaption,
        hashtags: hashtags
      });
      alert('수정되었습니다.');
      setIsEditing(false);
      // 부모 컴포넌트 데이터 갱신 (Invalidate queries)
      queryClient.invalidateQueries({ queryKey: ['feed'] });
      queryClient.invalidateQueries({ queryKey: ['search'] });
      queryClient.invalidateQueries({ queryKey: ['post', post.id] });
      // 현재 포스트 객체도 업데이트 (로컬 상태 반영은 어려우므로 일단 창 닫기 혹은 새로고침)
      onClose();
    } catch (e) {
      alert('수정 실패');
    }
  };

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) return;
    try {
      const res = await client.post('/collections', { name: newCollectionName });
      setNewCollectionName('');
      
      // 목록 새로고침 완료 대기
      await queryClient.invalidateQueries({ queryKey: ['collections'] });
      
      // 생성 후 바로 저장 로직 실행
      if (res.data?.id) {
        handleSaveToCollection(res.data.id);
      }
    } catch (e: any) { 
      console.error(e);
      if (e.response?.status === 400) {
        alert(e.response.data.detail || '이미 존재하는 폴더 이름입니다.');
      } else {
        alert('폴더 생성에 실패했습니다.');
      }
    }
  };

  const handleSaveToCollection = async (collectionId: string) => {
    try {
      const res = await client.post(`/collections/${collectionId}/items`, { post_id: post.id });
      setSaveStatus(res.data.status === 'already_exists' ? '이미 저장됨' : '저장되었습니다.');
      setIsSaved(true);
      
      // 목록 및 저장 상태 새로고침
      await queryClient.invalidateQueries({ queryKey: ['collections'] });
      await queryClient.invalidateQueries({ queryKey: ['savedIds'] });
      
    } catch (e) { setSaveStatus('저장 실패'); }
    setTimeout(() => setSaveStatus(''), 2000);
  };

  const handleUnsave = async () => {
    try {
      await client.delete(`/collections/items/${post.id}`);
      setIsSaved(false);
      setShowSaveMenu(false);
      setSaveStatus('저장이 취소되었습니다.');
      
      // 목록 새로고침
      await queryClient.invalidateQueries({ queryKey: ['collections'] });
      await queryClient.invalidateQueries({ queryKey: ['savedIds'] });
      
      setTimeout(() => setSaveStatus(''), 2000);
    } catch (e) { alert('저장 취소 실패'); }
  };

  const handleBookmarkClick = () => {
    if (!user) return alert('로그인이 필요합니다.');
    if (isSaved) {
      handleUnsave();
    } else {
      setShowSaveMenu(!showSaveMenu);
    }
  };

  // Derived Stats
  const isLiked = stats?.is_liked ?? post.is_liked;
  const likesCount = stats?.likes ?? post.like_count ?? 0;
  const safeComments = comments || [];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="modal-overlay" onClick={onClose}>
      <motion.div 
        initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
        className="post-modal-content" onClick={e => e.stopPropagation()} 
        style={{ backgroundColor: 'white', width: '100%', maxWidth: '1100px', height: 'min(90vh, 800px)', borderRadius: '32px', overflow: 'hidden', display: 'flex', boxShadow: 'var(--shadow-lg)' }}
      >
        <div style={{ flex: '1.2', backgroundColor: '#000', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <img src={post.content?.media_list?.find((m:any) => m.type === 'image')?.url} alt="" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
          {post.content?.media_list?.some((m: any) => m.type === 'audio') && (
            <>
              <button onClick={() => setIsMuted(!isMuted)} style={{ position: 'absolute', bottom: '1.5rem', left: '1.5rem', backgroundColor: 'rgba(255,255,255,0.2)', backdropFilter: 'blur(12px)', color: 'white', padding: '0.75rem', borderRadius: '50%', zIndex: 10 }}>{isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}</button>
              <audio ref={audioRef} src={post.content.media_list.find((m: any) => m.type === 'audio')?.url} autoPlay loop muted={isMuted} />
              <div style={{ position: 'absolute', bottom: 0, left: 0, width: '100%', height: '4px', backgroundColor: 'rgba(255,255,255,0.1)' }}><div style={{ height: '100%', width: `${progress}%`, backgroundColor: 'var(--primary-red)', transition: 'width 0.1s linear' }} /></div>
            </>
          )}
        </div>

        <div style={{ flex: '0.8', display: 'flex', flexDirection: 'column', minWidth: '350px', position: 'relative' }}>
          <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ width: '2.5rem', height: '2.5rem', borderRadius: '50%', backgroundColor: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                {post.author?.profile_image ? (
                  <img src={post.author.profile_image} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <span style={{ fontWeight: 800, color: 'var(--primary-red)' }}>{post.author?.nickname?.[0]}</span>
                )}
              </div>
              <span style={{ fontWeight: 800, fontSize: '0.9375rem' }}>{post.author?.nickname || '유저'}</span>
            </div>
            <button onClick={onClose} style={{ color: '#94a3b8', padding: '0.5rem' }}><X size={20} /></button>
          </div>

          <div className="no-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '1.5rem' }}>
            <div style={{ marginBottom: '2rem' }}>
              {isEditing ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <textarea 
                    value={editCaption} 
                    onChange={e => setEditCaption(e.target.value)}
                    placeholder="설명을 입력하세요..."
                    style={{ width: '100%', minHeight: '100px', padding: '1rem', borderRadius: '12px', border: '1px solid var(--primary-red)', outline: 'none', resize: 'none', fontWeight: 600 }}
                  />
                  <input 
                    value={editHashtags}
                    onChange={e => setEditHashtags(e.target.value)}
                    placeholder="해시태그 (쉼표 구분)"
                    style={{ width: '100%', padding: '0.75rem 1rem', borderRadius: '12px', border: '1px solid var(--primary-red)', outline: 'none', fontWeight: 600 }}
                  />
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button onClick={handleUpdatePost} style={{ flex: 1, backgroundColor: 'var(--black)', color: 'white', padding: '0.75rem', borderRadius: '8px', fontWeight: 800 }}>저장</button>
                    <button onClick={() => setIsEditing(false)} style={{ flex: 1, backgroundColor: '#f1f5f9', color: '#1a1a1a', padding: '0.75rem', borderRadius: '8px', fontWeight: 800 }}>취소</button>
                  </div>
                </div>
              ) : (
                <>
                  <p style={{ fontSize: '1rem', fontWeight: 600, lineHeight: 1.75, fontFamily: "'Noto Serif KR', Georgia, serif", color: '#1a1a1a', letterSpacing: '-0.01em' }}>{post.caption}</p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.875rem' }}>{post.hashtags?.map((h: any) => <span key={h.tag} style={{ color: '#6366f1', fontSize: '0.8125rem', fontWeight: 700, fontFamily: "'Noto Serif KR', sans-serif" }}>#{h.tag}</span>)}</div>
                </>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {safeComments.map((c: any, i: number) => (
                <div key={i} style={{ display: 'flex', gap: '0.75rem' }}>
                  <div style={{ width: '2rem', height: '2rem', borderRadius: '50%', backgroundColor: '#f8fafc', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                    {c.author?.profile_image ? (
                      <img src={c.author.profile_image} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                      <span style={{ fontSize: '0.625rem', fontWeight: 800 }}>{c.author?.nickname?.[0]}</span>
                    )}
                  </div>
                  <div style={{ flex: 1 }}><p style={{ fontSize: '0.875rem', lineHeight: 1.4 }}><span style={{ fontWeight: 800, marginRight: '0.5rem' }}>{c.author?.nickname || '유저'}</span>{c.content}</p><p style={{ fontSize: '0.75rem', color: '#cbd5e1', marginTop: '0.25rem' }}>{new Date(c.created_at).toLocaleDateString()}</p></div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ padding: '1.25rem 1.5rem', borderTop: '1px solid var(--border-color)', position: 'relative' }}>
            <div style={{ display: 'flex', gap: '1.25rem', marginBottom: '0.75rem', alignItems: 'center' }}>
              <motion.button whileTap={{ scale: 0.8 }} onClick={handleToggleLike} style={{ color: isLiked ? 'var(--primary-red)' : 'var(--text-main)', border: 'none', backgroundColor: 'transparent', padding: 0 }}><Heart size={28} fill={isLiked ? 'var(--primary-red)' : 'none'} strokeWidth={2.5} /></motion.button>
              
              {(user?.id === post.user_id || user?.user_id === post.user_id) && (
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <button onClick={() => setIsEditing(!isEditing)} style={{ color: isEditing ? 'var(--primary-red)' : '#94a3b8', border: 'none', backgroundColor: 'transparent', padding: 0 }} title="게시물 수정">
                    <Edit3 size={24} strokeWidth={2.5} />
                  </button>
                  <button onClick={handleDeletePost} style={{ color: '#94a3b8', border: 'none', backgroundColor: 'transparent', padding: 0 }} title="게시물 삭제">
                    <Trash2 size={24} strokeWidth={2.5} />
                  </button>
                </div>
              )}

              <div style={{ marginLeft: 'auto', position: 'relative' }}>
                <button onClick={handleBookmarkClick} style={{ color: isSaved || showSaveMenu ? 'var(--primary-red)' : 'var(--text-main)', border: 'none', backgroundColor: 'transparent', padding: 0 }}><Bookmark size={28} strokeWidth={2.5} fill={isSaved || showSaveMenu ? 'var(--primary-red)' : 'none'} /></button>
                <AnimatePresence>
                  {showSaveMenu && (
                    <motion.div 
                      ref={saveMenuRef}
                      key="save-menu" 
                      initial={{ opacity: 0, y: 10 }} 
                      animate={{ opacity: 1, y: 0 }} 
                      exit={{ opacity: 0, y: 10 }} 
                      style={{ position: 'absolute', bottom: '100%', right: 0, marginBottom: '0.5rem', width: '240px', backgroundColor: 'white', borderRadius: '16px', boxShadow: 'var(--shadow-lg)', border: '1px solid var(--border-color)', padding: '1rem', zIndex: 50 }}
                    >
                      <h4 style={{ fontSize: '0.875rem', fontWeight: 800, marginBottom: '0.75rem', color: 'var(--text-muted)' }}>내 폴더에 저장</h4>
                      <div style={{ maxHeight: '150px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1rem' }}>
                        {collections?.map((col: any) => <button key={col.id} onClick={() => handleSaveToCollection(col.id)} style={{ textAlign: 'left', padding: '0.75rem', borderRadius: '8px', fontSize: '0.875rem', fontWeight: 700, backgroundColor: '#f8fafc', display: 'flex', justifyContent: 'space-between' }}>{col.name}</button>)}
                      </div>
                      <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <input value={newCollectionName} onChange={e => setNewCollectionName(e.target.value)} placeholder="새 폴더" style={{ flex: 1, padding: '0.5rem', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.8125rem' }} />
                        <button onClick={handleCreateCollection} disabled={!newCollectionName.trim()} style={{ backgroundColor: 'var(--black)', color: 'white', padding: '0.5rem', borderRadius: '8px' }}><FolderPlus size={16} /></button>
                      </div>
                      {saveStatus && <p style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--primary-red)', marginTop: '0.75rem', textAlign: 'center' }}>{saveStatus}</p>}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
            <p style={{ fontWeight: 800, fontSize: '0.9375rem', marginBottom: '1.25rem' }}>좋아요 {likesCount}개</p>
            <form onSubmit={handleAddComment} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', backgroundColor: '#f8fafc', padding: '0.5rem 1rem', borderRadius: '16px' }}>
              <input value={newComment} onChange={e => setNewComment(e.target.value)} placeholder="댓글 달기..." style={{ flex: 1, border: 'none', backgroundColor: 'transparent', outline: 'none', fontSize: '0.875rem', fontWeight: 500 }} />
              <button type="submit" disabled={!newComment.trim()} style={{ color: 'var(--primary-red)' }}><Send size={20} strokeWidth={2.5} /></button>
            </form>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default PostModal;
