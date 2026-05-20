import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, Search, Home, Compass, User, LogOut, Heart, Bookmark, Settings as SettingsIcon, Music, Wand2, ChevronLeft, Camera, MoreVertical, Edit3, Trash2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './context/AuthContext';
import AuthModal from './components/AuthModal';
import PostModal from './components/PostModal';
import CreateModal from './components/CreateModal';
import SettingsModal from './components/SettingsModal';
import ImageSearchModal from './components/ImageSearchModal';
import client from './api/client';
import { useFeed, useSearch, useSavedIds, useCollections, useImageSearch } from './hooks/usePostQueries';

const SkeletonCard: React.FC = () => (
  <div className="masonry-item" style={{ breakInside: 'avoid', marginBottom: '1.5rem', borderRadius: '24px', overflow: 'hidden', backgroundColor: '#f1f5f9', height: `${Math.random() * 200 + 200}px`, position: 'relative' }}>
    <motion.div 
      initial={{ opacity: 0.3 }} 
      animate={{ opacity: 0.6 }} 
      transition={{ repeat: Infinity, duration: 1.5, ease: 'easeInOut' }}
      style={{ width: '100%', height: '100%', backgroundColor: '#e2e8f0' }} 
    />
  </div>
);

const AppContent: React.FC = () => {
  const { user, isLoggedIn, logout } = useAuth();
  const [viewMode, setViewMode] = useState<'all' | 'liked' | 'saved' | 'explore' | 'profile'>('all');
  const [profileTab, setProfileTab] = useState<'posts' | 'folders' | 'liked' | 'settings'>('posts');
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);
  const [editingCollectionId, setEditingCollectionId] = useState<string | null>(null);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [selectedCollection, setSelectedCollection] = useState<any | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchImage, setSearchImage] = useState<File | null>(null);

  useEffect(() => {
    if (viewMode !== 'explore') {
      setSearchQuery('');
      setSearchImage(null);
    }
  }, [viewMode]);

  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showImageSearchModal, setShowImageSearchModal] = useState(false);
  const [selectedPost, setSelectedPost] = useState<any | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const { checkAuth } = useAuth();
  const qc = useQueryClient();

  // React Query Hooks
  const { 
    data: infiniteFeedData, 
    isLoading: feedLoading, 
    fetchNextPage, 
    hasNextPage, 
    isFetchingNextPage 
  } = useFeed(
    viewMode === 'profile' 
      ? (profileTab === 'liked' ? 'liked' : (profileTab === 'folders' ? 'saved' : 'profile')) 
      : viewMode, 
    user?.id || user?.user_id || '0', 
    { collectionId: selectedCollection?.id }
  );

  const observerTarget = React.useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { threshold: 0.1 }
    );

    if (observerTarget.current) {
      observer.observe(observerTarget.current);
    }

    return () => {
      if (observerTarget.current) {
        observer.unobserve(observerTarget.current);
      }
    };
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  const feedPosts = infiniteFeedData?.pages.flatMap(page => page.posts) || [];
  const { data: searchResults, isFetching: searchLoading } = useSearch(searchQuery);
  const { data: imageSearchResults, isFetching: imageSearchLoading } = useImageSearch(searchImage, user?.id || user?.user_id);
  const { data: savedPostIds = [] } = useSavedIds(isLoggedIn);
  const { data: collections = [] } = useCollections(isLoggedIn && (viewMode === 'saved' || (viewMode === 'profile' && profileTab === 'folders')));

  const handleDeleteCollection = async (id: string) => {
    if (!window.confirm('이 폴더를 삭제하시겠습니까? 안의 아이템들은 사라지지만 원본 게시물은 유지됩니다.')) return;
    try {
      await client.delete(`/collections/${id}`);
      qc.invalidateQueries({ queryKey: ['collections'] });
      setActiveMenuId(null);
    } catch (err) {
      console.error('Failed to delete collection:', err);
    }
  };

  const handleRenameCollection = async (id: string) => {
    if (!newCollectionName.trim()) return;
    try {
      await client.patch(`/collections/${id}`, { name: newCollectionName });
      qc.invalidateQueries({ queryKey: ['collections'] });
      setEditingCollectionId(null);
      setNewCollectionName('');
      setActiveMenuId(null);
    } catch (err) {
      console.error('Failed to rename collection:', err);
    }
  };

  const handleCreateCollection = async () => {
    const name = window.prompt('새 폴더 이름을 입력하세요');
    if (!name) return;
    try {
      await client.post('/collections/', { name });
      qc.invalidateQueries({ queryKey: ['collections'] });
    } catch (e) { alert('폴더 생성 실패'); }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      await client.post('/users/me/image', formData);
      await checkAuth();
    } catch (err) { alert('이미지 업로드 실패'); }
    finally { setIsUploading(false); }
  };

  useEffect(() => {
    if (!isLoggedIn && (viewMode === 'profile' || viewMode === 'saved')) {
      setViewMode('all');
      setShowAuthModal(true);
    }
  }, [viewMode, isLoggedIn]);

  const displayPosts = (viewMode === 'explore') 
    ? (searchImage ? (imageSearchResults || []) : (searchQuery.trim() ? (searchResults || []) : feedPosts)) 
    : feedPosts;
  const isLoading = (viewMode === 'explore')
    ? (searchImage ? imageSearchLoading : (searchQuery.trim() ? searchLoading : (feedLoading && feedPosts.length === 0)))
    : (feedLoading && feedPosts.length === 0);

  // Masonry Column Distribution Logic
  const [columnCount, setColumnCount] = useState(2);
  useEffect(() => {
    const updateColumnCount = () => {
      const width = window.innerWidth;
      if (width >= 1536) setColumnCount(6);
      else if (width >= 1280) setColumnCount(5);
      else if (width >= 1024) setColumnCount(4);
      else if (width >= 768) setColumnCount(3);
      else setColumnCount(2);
    };
    updateColumnCount();
    window.addEventListener('resize', updateColumnCount);
    return () => window.removeEventListener('resize', updateColumnCount);
  }, []);

  const columns: any[][] = Array.from({ length: columnCount }, () => []);
  displayPosts.forEach((post, index) => {
    columns[index % columnCount].push(post);
  });

  const ProfileHeader = () => (
    <div style={{ maxWidth: '935px', margin: '3rem auto 0', display: 'flex', flexDirection: 'column', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'start', gap: '3rem', padding: '0 1rem', marginBottom: '3rem' }}>
        <div 
          onClick={() => fileInputRef.current?.click()}
          style={{ position: 'relative', width: '150px', height: '150px', borderRadius: '50%', overflow: 'hidden', backgroundColor: '#f1f5f9', flexShrink: 0, cursor: 'pointer', border: '1px solid #dbdbdb' }}
        >
          {isUploading ? (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}><Plus size={32} /></motion.div>
            </div>
          ) : user?.profile_image ? (
            <img src={user.profile_image} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '3rem', fontWeight: 900, color: '#4f46e5' }}>{user?.nickname?.[0]}</div>
          )}
          <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.3)', opacity: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'opacity 0.2s', color: 'white' }} 
            onMouseEnter={e => e.currentTarget.style.opacity = '1'} onMouseLeave={e => e.currentTarget.style.opacity = '0'}
          >
            <Camera size={24} />
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginBottom: '1.25rem' }}>
            <h2 style={{ fontSize: '1.75rem', fontWeight: 300, color: '#262626' }}>{user?.nickname}</h2>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <button 
                onClick={() => setShowSettingsModal(true)} 
                style={{ border: '1px solid #dbdbdb', borderRadius: '8px', padding: '6px 16px', fontSize: '14px', fontWeight: 600, backgroundColor: 'white', cursor: 'pointer' }}>
                프로필 편집
              </button>
              <button 
                onClick={logout}
                style={{ border: '1px solid #ff4d4f', color: '#ff4d4f', borderRadius: '8px', padding: '6px 16px', fontSize: '14px', fontWeight: 600, backgroundColor: 'white', cursor: 'pointer' }}>
                로그아웃
              </button>
            </div>
          </div>
          <div style={{ whiteSpace: 'pre-wrap', marginTop: '0.5rem', fontSize: '1rem', lineHeight: '1.5' }}>{user?.bio || '나만의 무드를 기록해보세요.'}</div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', borderTop: '1px solid #dbdbdb', gap: '4rem', marginBottom: '2rem' }}>
        <button onClick={() => setProfileTab('posts')} style={{ display: 'flex', alignItems: 'center', marginTop: '-1px', padding: '1.25rem 0.5rem', fontSize: '15px', fontWeight: profileTab === 'posts' ? 800 : 500, color: profileTab === 'posts' ? 'black' : '#8e8e8e', backgroundColor: 'transparent', border: 'none', cursor: 'pointer' }}>게시물</button>
        <button onClick={() => setProfileTab('folders')} style={{ display: 'flex', alignItems: 'center', marginTop: '-1px', padding: '1.25rem 0.5rem', fontSize: '15px', fontWeight: profileTab === 'folders' ? 800 : 500, color: profileTab === 'folders' ? 'black' : '#8e8e8e', backgroundColor: 'transparent', border: 'none', cursor: 'pointer' }}>폴더 (저장됨)</button>
        <button onClick={() => setProfileTab('liked')} style={{ display: 'flex', alignItems: 'center', marginTop: '-1px', padding: '1.25rem 0.5rem', fontSize: '15px', fontWeight: profileTab === 'liked' ? 800 : 500, color: profileTab === 'liked' ? 'black' : '#8e8e8e', backgroundColor: 'transparent', border: 'none', cursor: 'pointer' }}>좋아요</button>
      </div>
    </div>
  );

  return (
    <div className="app-container">
      <nav style={{ position: 'sticky', top: 0, zIndex: 50, backgroundColor: 'rgba(255, 255, 255, 0.85)', backdropFilter: 'blur(20px)', borderBottom: '1px solid var(--border-color)', padding: '0.75rem 2.5rem' }}>
        <div style={{ maxWidth: '1920px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '3rem' }}>
            <motion.h1 whileHover={{ scale: 1.05 }} onClick={() => { setViewMode('all'); setSelectedCollection(null); }} style={{ fontSize: '1.75rem', fontWeight: 900, color: 'var(--primary-red)', letterSpacing: '-0.06em', cursor: 'pointer' }}>Interest</motion.h1>
            <div style={{ display: 'flex', gap: '2rem', fontWeight: 800, fontSize: '0.9375rem' }}>
              <button onClick={() => { setViewMode('all'); setSelectedCollection(null); }} style={{ color: viewMode === 'all' ? 'var(--primary-red)' : 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.625rem' }}><Home size={20} /> 홈</button>
              <button onClick={() => setViewMode('explore')} style={{ color: viewMode === 'explore' ? 'var(--primary-red)' : 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.625rem' }}><Compass size={20} /> 탐색</button>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
            {!isLoggedIn ? (
              <button onClick={() => setShowAuthModal(true)} style={{ backgroundColor: 'var(--primary-red)', color: 'white', padding: '0.75rem 1.5rem', fontWeight: 800, borderRadius: '999px' }}>로그인</button>
            ) : (
                      <div style={{ position: 'relative' }}>
                <button 
                  onClick={() => { setViewMode('profile'); setProfileTab('posts'); }}
                  style={{ width: '2.75rem', height: '2.75rem', borderRadius: '50%', backgroundColor: '#eef2ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 900, color: '#4f46e5', overflow: 'hidden', border: viewMode === 'profile' ? '2px solid var(--primary-red)' : 'none' }}>
                  {user?.profile_image ? <img src={user.profile_image} style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : user?.nickname?.[0]}
                </button>
              </div>
            )}
          </div>
        </div>
      </nav>

      <main style={{ maxWidth: '1920px', margin: '0 auto' }}>
        {viewMode === 'explore' && (
          <div style={{ padding: '2rem 1rem', display: 'flex', justifyContent: 'center' }}>
            <div style={{ position: 'relative', width: '100%', maxWidth: '600px' }}>
              {searchImage ? (
                <div style={{ position: 'absolute', left: '1.5rem', top: '50%', transform: 'translateY(-50%)', display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: 'white', padding: '0.35rem 0.75rem', borderRadius: '16px', border: '1px solid #e2e8f0', boxShadow: '0 2px 6px rgba(0,0,0,0.05)', zIndex: 10 }}>
                  <img src={URL.createObjectURL(searchImage)} style={{ width: '20px', height: '20px', borderRadius: '6px', objectFit: 'cover' }} />
                  <span style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--text-main)' }}>이미지 검색</span>
                  <button 
                    onClick={() => setSearchImage(null)} 
                    style={{ background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: 0, color: 'var(--text-muted)' }}
                  >
                    <Plus size={14} style={{ transform: 'rotate(45deg)' }} />
                  </button>
                </div>
              ) : (
                <Search size={20} style={{ position: 'absolute', left: '1.5rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              )}
              <input 
                autoFocus 
                placeholder={searchImage ? "" : "검색어를 입력해보세요"} 
                disabled={!!searchImage}
                value={searchImage ? "이미지 기준으로 추천 검색된 결과입니다." : searchQuery} 
                onChange={e => setSearchQuery(e.target.value)} 
                style={{ 
                  width: '100%', 
                  padding: `1rem 3.5rem 1rem ${searchImage ? '11rem' : '3.75rem'}`, 
                  borderRadius: '24px', 
                  backgroundColor: '#f1f5f9', 
                  border: 'none', 
                  fontSize: '1.1rem', 
                  fontWeight: 600, 
                  boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                  color: searchImage ? 'var(--text-muted)' : 'inherit'
                }} 
              />
              <button 
                onClick={() => setShowImageSearchModal(true)}
                style={{ position: 'absolute', right: '1.5rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
              >
                <Camera size={20} />
              </button>
            </div>
          </div>
        )}
        {viewMode === 'profile' && <ProfileHeader />}
        
        <AnimatePresence mode="wait">
          {isLoading ? (
            <div key="loading" style={{ display: 'flex', justifyContent: 'center', padding: '10rem' }}><motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }} style={{ width: '3rem', height: '3rem', border: '4px solid #f1f5f9', borderTopColor: 'var(--primary-red)', borderRadius: '50%' }} /></div>
          ) : viewMode === 'profile' && profileTab === 'settings' ? (
            <motion.div key="settings" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ padding: '0 1rem' }}>
              <div style={{ width: '100%', maxWidth: '600px', margin: '0 auto', animation: 'fadeIn 0.3s ease' }}>
                <SettingsForm onComplete={() => setProfileTab('posts')} />
              </div>
            </motion.div>
          ) : (viewMode === 'saved' || (viewMode === 'profile' && profileTab === 'folders')) && !selectedCollection ? (
            <motion.div key="collections" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ padding: '0 1rem' }}>
              <div style={{ maxWidth: '935px', margin: '0 auto', paddingBottom: '5rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '2rem' }}>
                  <motion.div 
                    whileHover={{ scale: 1.02 }} 
                    onClick={async () => {
                      const name = window.prompt('새 폴더 이름을 입력하세요');
                      if (name) {
                        try {
                          await client.post('/collections/', { name });
                          await qc.invalidateQueries({ queryKey: ['collections'] });
                        } catch (e) { alert('폴더 생성 실패'); }
                      }
                    }} 
                    style={{ height: '200px', borderRadius: '24px', border: '2px dashed #dbdbdb', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', color: '#8e8e8e' }}
                  >
                    <Plus size={32} />
                    <span style={{ marginTop: '0.5rem', fontWeight: 600 }}>새 폴더 만들기</span>
                  </motion.div>
                  {collections.map((col: any) => (
                    <motion.div key={col.id} whileHover={{ y: -5 }} style={{ cursor: 'pointer', position: 'relative' }}>
                      <div onClick={() => setSelectedCollection(col)}>
                        <div style={{ height: '200px', borderRadius: '24px', overflow: 'hidden', display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2px', backgroundColor: '#f1f5f9', border: '1px solid #e2e8f0' }}>
                          <div style={{ backgroundImage: col.collection_images?.[0] ? `url("${col.collection_images[0]}")` : 'none', backgroundColor: '#e2e8f0', backgroundSize: 'cover', backgroundPosition: 'center' }}>
                            {!col.collection_images?.[0] && <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Bookmark size={32} color="#cbd5e1" /></div>}
                          </div>
                          <div style={{ display: 'grid', gridTemplateRows: '1fr 1fr', gap: '2px' }}>
                            <div style={{ backgroundImage: col.collection_images?.[1] ? `url("${col.collection_images[1]}")` : 'none', backgroundColor: '#f1f5f9', backgroundSize: 'cover', backgroundPosition: 'center' }} />
                            <div style={{ backgroundImage: col.collection_images?.[2] ? `url("${col.collection_images[2]}")` : 'none', backgroundColor: '#f8fafc', backgroundSize: 'cover', backgroundPosition: 'center' }} />
                          </div>
                        </div>
                        <p style={{ fontWeight: 800, marginTop: '1rem', fontSize: '1.1rem' }}>{col.name}</p>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>{col.item_count || 0}개의 무드</p>
                      </div>
                      
                      <button 
                        onClick={(e) => { e.stopPropagation(); setActiveMenuId(activeMenuId === col.id ? null : col.id); }}
                        style={{ position: 'absolute', top: '12px', right: '12px', padding: '8px', borderRadius: '50%', backgroundColor: 'rgba(255,255,255,0.9)', border: 'none', cursor: 'pointer', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                      >
                        <MoreVertical size={16} color="#262626" />
                      </button>

                      <AnimatePresence>
                        {activeMenuId === col.id && (
                          <motion.div 
                            initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
                            style={{ position: 'absolute', top: '50px', right: '12px', width: '160px', backgroundColor: 'white', borderRadius: '16px', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.15)', border: '1px solid #f1f5f9', zIndex: 10, padding: '0.5rem' }}
                          >
                            <button 
                              onClick={(e) => { e.stopPropagation(); setEditingCollectionId(col.id); setNewCollectionName(col.name); }}
                              style={{ width: '100%', padding: '0.75rem', borderRadius: '10px', border: 'none', backgroundColor: 'transparent', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.875rem', cursor: 'pointer', textAlign: 'left' }}
                              className="folder-menu-item"
                            >
                              <Edit3 size={14} /> 이름 변경
                            </button>
                            <button 
                              onClick={(e) => { e.stopPropagation(); handleDeleteCollection(col.id); }}
                              style={{ width: '100%', padding: '0.75rem', borderRadius: '10px', border: 'none', backgroundColor: 'transparent', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.875rem', cursor: 'pointer', textAlign: 'left', color: '#ff4d4f' }}
                              className="folder-menu-item"
                            >
                              <Trash2 size={14} /> 삭제
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>

                      {editingCollectionId === col.id && (
                        <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: '24px', zIndex: 11, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '1.5rem', gap: '1rem' }} onClick={e => e.stopPropagation()}>
                          <input 
                            autoFocus
                            value={newCollectionName}
                            onChange={e => setNewCollectionName(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && handleRenameCollection(col.id)}
                            style={{ width: '100%', padding: '0.75rem 1rem', borderRadius: '12px', border: '2px solid var(--primary-red)', outline: 'none', fontWeight: 700 }}
                          />
                          <div style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
                            <button onClick={() => handleRenameCollection(col.id)} style={{ flex: 1, backgroundColor: 'var(--black)', color: 'white', border: 'none', padding: '0.6rem', borderRadius: '10px', fontWeight: 800, cursor: 'pointer' }}>저장</button>
                            <button onClick={() => setEditingCollectionId(null)} style={{ flex: 1, backgroundColor: '#f1f5f9', color: '#1a1a1a', border: 'none', padding: '0.6rem', borderRadius: '10px', fontWeight: 800, cursor: 'pointer' }}>취소</button>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div key="feed" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="masonry-container" style={{ padding: '0 1rem' }}>
              {((viewMode === 'saved' || (viewMode === 'profile' && profileTab === 'folders')) && selectedCollection) && (
                <div style={{ gridColumn: '1/-1', marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <button onClick={() => setSelectedCollection(null)} style={{ width: '3rem', height: '3rem', borderRadius: '50%', backgroundColor: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><ChevronLeft /></button>
                  <h2 style={{ fontSize: '2rem', fontWeight: 900 }}>{selectedCollection.name}</h2>
                </div>
              )}
              
              {isLoading && displayPosts.length === 0 ? (
                Array.from({ length: 12 }).map((_, i) => (
                  <SkeletonCard key={`skeleton-${i}`} />
                ))
              ) : (
                columns.map((column, colIndex) => (
                  <div key={`col-${colIndex}`} className="masonry-column">
                    {column.map((post: any) => (
                      <motion.div 
                        key={post.id} 
                        whileHover={{ scale: 1.02 }} 
                        onClick={() => setSelectedPost(post)} 
                        className="masonry-item" 
                        style={{ borderRadius: '24px', overflow: 'hidden', position: 'relative', cursor: 'zoom-in', boxShadow: 'var(--shadow-sm)' }}
                      >
                        <img 
                          src={post.content?.media_list?.find((m:any)=>m.type==='image')?.url} 
                          style={{ width: '100%', display: 'block' }} 
                        />
                        <div style={{ position: 'absolute', bottom: '1rem', left: '1rem', display: 'flex', gap: '0.5rem' }}>
                          {(() => {
                            const isAI = post.content?.is_ai;
                            const hasAudio = post.content?.media_list?.some((m:any)=>m.type==='audio');
                            if (isAI) return <div style={{ backgroundColor: 'rgba(0,0,0,0.7)', color: 'white', padding: '0.4rem 0.8rem', borderRadius: '99px', fontSize: '0.7rem', fontWeight: 900, display: 'flex', alignItems: 'center', gap: '0.3rem' }}><Wand2 size={12}/> ai</div>;
                            if (hasAudio) return <div style={{ backgroundColor: 'rgba(255,255,255,0.8)', padding: '0.4rem 0.8rem', borderRadius: '99px', fontSize: '0.7rem', fontWeight: 900, display: 'flex', alignItems: 'center', gap: '0.3rem' }}><Music size={12}/> music</div>;
                            return null;
                          })()}
                        </div>
                        {Array.isArray(savedPostIds) && savedPostIds.some((id: any) => String(id) === String(post.id)) && (
                          <div style={{ position: 'absolute', top: '1rem', right: '1rem', backgroundColor: 'var(--primary-red)', color: 'white', padding: '0.5rem', borderRadius: '50%', boxShadow: 'var(--shadow-md)' }}>
                            <Bookmark size={16} fill="white" />
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                ))
              )}
              <div ref={observerTarget} style={{ height: '100px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'absolute', bottom: 0 }}>
                {isFetchingNextPage && <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }} style={{ width: '2rem', height: '2rem', border: '3px solid #f1f5f9', borderTopColor: 'var(--primary-red)', borderRadius: '50%' }} />}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <motion.button whileHover={{ scale: 1.1, rotate: 90 }} onClick={() => isLoggedIn ? setShowCreateModal(true) : setShowAuthModal(true)} style={{ position: 'fixed', bottom: '2.5rem', right: '2.5rem', width: '4rem', height: '4rem', backgroundColor: 'var(--primary-red)', color: 'white', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: 'var(--shadow-lg)', zIndex: 40 }}><Plus size={36} strokeWidth={3} /></motion.button>

      <AnimatePresence>
        {showAuthModal && <AuthModal onClose={() => setShowAuthModal(false)} />}
        {showCreateModal && (
          <CreateModal 
            onClose={() => setShowCreateModal(false)} 
            onSuccess={() => {
              qc.invalidateQueries({ queryKey: ['feed'] });
              qc.invalidateQueries({ queryKey: ['search'] });
              qc.invalidateQueries({ queryKey: ['imageSearch'] });
              qc.invalidateQueries({ queryKey: ['collections'] });
            }} 
          />
        )}
        {showSettingsModal && <SettingsModal onClose={() => setShowSettingsModal(false)} />}
        {showImageSearchModal && (
          <ImageSearchModal 
            onClose={() => setShowImageSearchModal(false)} 
            onSelectImage={(file) => {
              setSearchQuery('');
              setSearchImage(file);
            }} 
          />
        )}
        {selectedPost && (
          <PostModal 
            post={selectedPost} 
            isSavedInitial={Array.isArray(savedPostIds) && savedPostIds.some((id: any) => String(id) === String(selectedPost.id))} 
            onClose={() => { 
              setSelectedPost(null); 
              qc.invalidateQueries({ queryKey: ['savedIds'] }); 
            }} 
            onDelete={() => {
              qc.invalidateQueries({ queryKey: ['feed'] });
              qc.invalidateQueries({ queryKey: ['search'] });
              qc.invalidateQueries({ queryKey: ['collections'] });
            }} 
          />
        )}
      </AnimatePresence>
      <input type="file" ref={fileInputRef} onChange={handleImageUpload} style={{ display: 'none' }} accept="image/*" />
      <style>{`.menu-item:hover { background-color: #f8fafc; }`}</style>
    </div>
  );
};

const SettingsForm: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
  const { user, checkAuth } = useAuth();
  const [nickname, setNickname] = useState(user?.nickname || '');
  const [bio, setBio] = useState(user?.bio || '');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await client.put('/users/me', { nickname, bio });
      await checkAuth();
      alert('프로필이 업데이트되었습니다.');
      onComplete();
    } catch (err) { alert('업데이트 실패'); }
    finally { setLoading(false); }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '2rem' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <label style={{ fontSize: '0.875rem', fontWeight: 700, color: '#262626' }}>사용자 이름</label>
        <input value={nickname} onChange={e => setNickname(e.target.value)} placeholder="닉네임" style={{ border: '1px solid #dbdbdb' }} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <label style={{ fontSize: '0.875rem', fontWeight: 700, color: '#262626' }}>소개글</label>
        <textarea value={bio} onChange={e => setBio(e.target.value)} placeholder="자신을 표현해보세요." style={{ width: '100%', minHeight: '100px', border: '1px solid #dbdbdb', borderRadius: '12px', padding: '12px 16px', fontFamily: 'inherit', resize: 'vertical' }} />
      </div>
      <button type="submit" disabled={loading} style={{ backgroundColor: 'var(--primary-red)', color: 'white', padding: '12px', fontWeight: 800, fontSize: '0.9375rem', marginTop: '1rem' }}>
        {loading ? '저장 중...' : '프로필 저장'}
      </button>
    </form>
  );
};

const App: React.FC = () => (
  <AuthProvider>
    <AppContent />
  </AuthProvider>
);

export default App;
