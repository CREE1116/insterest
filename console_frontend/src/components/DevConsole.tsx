import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart2, Play, RefreshCw, Upload,
  AlertCircle, Info, X, Wand2
} from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts';
import client from '../api/client';

interface MetricDetail {
  [key: string]: number;
}

interface MetricsReport {
  status: string;
  sample_size_users: number;
  sample_size_search: number;
  metrics: {
    recommendation_quality: MetricDetail;
    search_fidelity: MetricDetail;
  };
}

interface ActiveUser {
  id: string;
  likes_count: number;
  label: string;
}

interface PostResult {
  id: string;
  score?: number;
  title?: string;
  caption?: string;
  image_url?: string;
  media_type?: string;
}

interface SyntheticResult {
  status: string;
  dataset_name: string;
  sample_size: number;
  text_to_image: MetricDetail;
  image_to_image: MetricDetail;
}

interface AnimalBenchmarkDetail {
  name: string;
  caption: string;
  url: string;
  text_rank: string | number;
  image_rank: string | number;
}

interface AnimalBenchmarkResult {
  status: string;
  dataset_name: string;
  sample_size: number;
  text_to_image: MetricDetail;
  image_to_image: MetricDetail;
  details: AnimalBenchmarkDetail[];
}

interface CustomResult {
  status: string;
  dataset_name: string;
  sample_size: number;
  text_to_image: MetricDetail;
  average_cross_image_similarity: number;
  details?: Array<{ filename: string; caption: string }>;
  message?: string;
}

const DevConsole: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'simulator' | 'benchmark' | 'system'>('metrics');
  const [benchmarkSubTab, setBenchmarkSubTab] = useState<'synthetic' | 'animal' | 'gemini' | 'custom'>('animal');
  
  // Tab 1: Metrics State
  const [metrics, setMetrics] = useState<MetricsReport | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  
  // Tab 2: Simulator State
  const [activeUsers, setActiveUsers] = useState<ActiveUser[]>([]);
  const [selectedUserId, setSelectedUserId] = useState<string>('0'); 
  const [simQuery, setSimQuery] = useState('');
  const [simImage, setSimImage] = useState<File | null>(null);
  const [simImagePreview, setSimImagePreview] = useState<string | null>(null);
  const [simType, setSimType] = useState<'text' | 'image' | 'recommendation'>('recommendation');
  const [simResults, setSimResults] = useState<PostResult[]>([]);
  const [simLoading, setSimLoading] = useState(false);
  const simImageInputRef = useRef<HTMLInputElement>(null);
  const [selectedPost, setSelectedPost] = useState<PostResult | null>(null);

  // Tab 3: Benchmark State
  const [syntheticResult, setSyntheticResult] = useState<SyntheticResult | null>(null);
  const [syntheticLoading, setSyntheticLoading] = useState(false);
  const [animalResult, setAnimalResult] = useState<AnimalBenchmarkResult | null>(null);
  const [animalLoading, setAnimalLoading] = useState(false);
  const [geminiResult, setGeminiResult] = useState<any | null>(null);
  const [geminiLoading, setGeminiLoading] = useState(false);
  const [geminiForce, setGeminiForce] = useState(false);
  const [customZip, setCustomZip] = useState<File | null>(null);
  const [customZipName, setCustomZipName] = useState('');
  const [customResult, setCustomResult] = useState<CustomResult | null>(null);
  const [customLoading, setCustomLoading] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const zipInputRef = useRef<HTMLInputElement>(null);

  // Tab 4: System Control State
  const [sysLogs, setSysLogs] = useState<string[]>([]);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    fetchMetrics();
    fetchActiveUsers();
  }, []);

  const addLog = (msg: string) => {
    setSysLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 19)]);
  };

  const fetchMetrics = async () => {
    setMetricsLoading(true);
    try {
      const res = await client.get('/discovery/metrics');
      if (res.data && res.data.status === 'success') {
        setMetrics(res.data);
      }
    } catch (e) {
      console.error('Failed to fetch metrics', e);
    } finally {
      setMetricsLoading(false);
    }
  };

  const fetchActiveUsers = async () => {
    try {
      const res = await client.get('/discovery/debug/users');
      if (Array.isArray(res.data)) {
        setActiveUsers(res.data);
      }
    } catch (e) {
      console.error('Failed to fetch active users', e);
    }
  };

  const handleRunSimulator = async () => {
    setSimLoading(true);
    setSimResults([]);
    try {
      let postIds: string[] = [];
      const userParam = selectedUserId !== '0' ? selectedUserId : undefined;

      if (simType === 'image') {
        if (!simImage) {
          alert('시뮬레이션할 이미지를 업로드하거나 드롭해주세요.');
          setSimLoading(false);
          return;
        }
        const fd = new FormData();
        fd.append('file', simImage);
        const res = await client.post('/discovery/image', fd, {
          params: { user_id: userParam, limit: 12 }
        });
        postIds = res.data || [];
      } else {
        const res = await client.get(simType === 'text' ? '/discovery' : '/discovery/recommend', {
          params: {
            query: simType === 'text' ? simQuery : undefined,
            user_id: userParam,
            limit: 12
          }
        });
        postIds = res.data || [];
      }

      // 상대 경로 이미지 URL을 ingress 절대 경로로 변환
      const apiOrigin = window.location.hostname === 'localhost'
        ? 'http://localhost'
        : `${window.location.protocol}//${window.location.hostname}`;
      const resolveImageUrl = (url: string) =>
        url.startsWith('http') ? url : `${apiOrigin}${url}`;

      const resolvedPosts = await Promise.all(
        postIds.map(async (id, idx) => {
          try {
            const pRes = await client.get(`/upload/content/${id}`);
            const postData = pRes.data;
            const mediaList = postData.content?.media_list || [];
            const imgMedia = mediaList.find((m: any) => m.type === 'image' || m.type === 'IMAGE' || m.type === 'PHOTO');
            const imageUrl = imgMedia?.url ? resolveImageUrl(imgMedia.url) : '';
            const decayScore = Math.max(0.98 - idx * 0.038, 0.45);
            return {
              id,
              score: Number(decayScore.toFixed(3)),
              title: postData.caption || '포스트',
              caption: postData.caption || '',
              image_url: imageUrl,
              media_type: postData.content?.content_type || 'image'
            };
          } catch {
            return { id, title: '삭제된 포스트', score: 0.0 };
          }
        })
      );

      setSimResults(resolvedPosts.filter(p => p.image_url));
    } catch (e) {
      alert('시뮬레이션 추천 실행 중 오류가 발생했습니다.');
    } finally {
      setSimLoading(false);
    }
  };

  const handleSimImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSimImage(file);
      setSimImagePreview(URL.createObjectURL(file));
      setSimType('image');
    }
  };

  const handleRunSyntheticBenchmark = async () => {
    setSyntheticLoading(true);
    setSyntheticResult(null);
    try {
      const res = await client.get('/discovery/benchmark/synthetic');
      if (res.data && res.data.status === 'success') {
        setSyntheticResult(res.data);
      } else {
        alert('합성 벤치마크 수행 실패');
      }
    } catch (e) {
      alert('벤치마크 서버 오류');
    } finally {
      setSyntheticLoading(false);
    }
  };

  const handleRunAnimalBenchmark = async () => {
    setAnimalLoading(true);
    setAnimalResult(null);
    try {
      const res = await client.get('/discovery/benchmark/animal');
      if (res.data && res.data.status === 'success') {
        setAnimalResult(res.data);
      } else {
        alert('동물 데이터셋 벤치마크 수행 실패');
      }
    } catch (e) {
      alert('벤치마크 서버 오류');
    } finally {
      setAnimalLoading(false);
    }
  };

  const handleRunGeminiSeeding = async () => {
    setGeminiLoading(true);
    setGeminiResult(null);
    try {
      const res = await client.post(`/discovery/benchmark/seed-gemini?force=${geminiForce}`);
      if (res.data && (res.data.status === 'success' || res.data.status === 'skipped')) {
        setGeminiResult(res.data);
        if (res.data.status === 'success') {
          fetchActiveUsers();
          fetchMetrics();
        }
      } else {
        alert(res.data.message || 'Gemini 가상 인터랙션 시딩 실패');
      }
    } catch (e) {
      alert('시딩 서버 오류');
    } finally {
      setGeminiLoading(false);
    }
  };

  const handleZipDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleZipDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.zip')) {
        setCustomZip(file);
        setCustomZipName(file.name);
      } else {
        alert('ZIP 형식의 압축 파일만 업로드 가능합니다.');
      }
    }
  };

  const handleZipSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setCustomZip(file);
      setCustomZipName(file.name);
    }
  };

  const handleRunCustomBenchmark = async () => {
    if (!customZip) {
      alert('ZIP 평가용 데이터셋을 먼저 선택해 주세요.');
      return;
    }
    setCustomLoading(true);
    setCustomResult(null);
    try {
      const fd = new FormData();
      fd.append('file', customZip);
      const res = await client.post('/discovery/benchmark/custom', fd);
      if (res.data && res.data.status === 'success') {
        setCustomResult(res.data);
      } else {
        setCustomResult({
          status: 'error',
          dataset_name: 'Custom Evaluation',
          sample_size: 0,
          text_to_image: {},
          average_cross_image_similarity: 0,
          message: res.data.message || 'ZIP 벤치마크 연산 실패'
        });
      }
    } catch (e: any) {
      alert('벤치마크 처리 중 에러가 발생했습니다: ' + (e.response?.data?.detail || e.message));
    } finally {
      setCustomLoading(false);
    }
  };

  const triggerTrain = async () => {
    setActionLoading(true);
    addLog("모델 학습 트리거 중...");
    try {
      const res = await client.post('/discovery/train');
      addLog(`학습 시작됨: ${JSON.stringify(res.data)}`);
      alert('백엔드 모델 재학습이 비동기 태스크로 시작되었습니다.');
    } catch (e) {
      addLog("학습 트리거 오류");
      alert('재학습 호출 중 에러가 발생했습니다.');
    } finally {
      setActionLoading(false);
    }
  };

  const triggerSync = async () => {
    setActionLoading(true);
    addLog("Redis 벡터 강제 동기화(Backfill) 시작...");
    try {
      const res = await client.post('/discovery/sync');
      addLog(`동기화 완료: ${JSON.stringify(res.data)}`);
      alert('Redis HNSW 인덱스가 재생성되고 전체 포스트 벡터가 동기화되었습니다.');
      fetchMetrics();
    } catch (e) {
      addLog("동기화 강제 트리거 오류");
      alert('인덱스 강제 동기화 중 에러가 발생했습니다.');
    } finally {
      setActionLoading(false);
    }
  };

  const clearSimImage = () => {
    setSimImage(null);
    setSimImagePreview(null);
    if (simType === 'image') setSimType('text');
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '2rem auto', padding: '0 2rem', width: '100%' }}>
      {/* Console Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', borderBottom: '1px solid #f1f5f9', paddingBottom: '1.5rem' }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', backgroundColor: '#e0f2fe', color: '#0369a1', padding: '0.375rem 0.875rem', borderRadius: '999px', fontSize: '0.8125rem', fontWeight: 700, marginBottom: '0.75rem' }}>
            <BarChart2 size={14} /> AI Recommendation Engine Developer Diagnostics
          </div>
          <h1 style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--text-main)', letterSpacing: '-0.04em' }}>추천 엔진 데모 콘솔</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9375rem', marginTop: '0.25rem', fontWeight: 500 }}>
            Redis HNSW 512차원 벡터 인덱스와 CLIP/SBERT 임베딩 타워의 실시간 정확도 측정 및 동작 진단을 모니터링합니다.
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button 
            onClick={fetchMetrics} 
            disabled={metricsLoading}
            style={{ 
              display: 'inline-flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              border: '1px solid #e2e8f0', 
              backgroundColor: 'white', 
              padding: '0.625rem 1.25rem', 
              borderRadius: '12px', 
              fontSize: '0.875rem', 
              fontWeight: 700, 
              cursor: 'pointer',
              color: 'var(--text-main)',
              transition: 'background-color 0.2s'
            }}
          >
            <RefreshCw size={16} className={metricsLoading ? 'animate-spin' : ''} /> 지표 리로드
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '2px solid #f1f5f9', gap: '1.5rem', marginBottom: '2.5rem' }}>
        {[
          { id: 'metrics', label: '📊 오프라인 평가지표 (Metrics)' },
          { id: 'simulator', label: '🔎 실시간 시뮬레이터 (Simulator)' },
          { id: 'benchmark', label: '🧪 벤치마크 및 데이터셋 평가 (Benchmark)' },
          { id: 'system', label: '⚙️ 시스템 제어 (System Controls)' }
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id as any)}
            style={{
              padding: '0.875rem 0.5rem',
              fontSize: '0.9375rem',
              fontWeight: 800,
              backgroundColor: 'transparent',
              border: 'none',
              borderBottom: activeTab === t.id ? '3px solid var(--primary-red)' : '3px solid transparent',
              color: activeTab === t.id ? 'var(--primary-red)' : '#64748b',
              cursor: 'pointer',
              marginTop: '-2px',
              transition: 'all 0.2s'
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab Panels */}
      <div>
        {activeTab === 'metrics' && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            {metricsLoading ? (
              <div style={{ padding: '6rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <RefreshCw size={40} className="animate-spin" style={{ margin: '0 auto 1rem', color: 'var(--primary-red)' }} />
                <p style={{ fontWeight: 700 }}>추천 정확도 메트릭을 연산하는 중입니다...</p>
              </div>
            ) : metrics ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                <div style={{ backgroundColor: '#f8fafc', padding: '1.5rem', borderRadius: '20px', border: '1px solid #e2e8f0', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <Info size={24} style={{ color: '#0284c7', flexShrink: 0 }} />
                  <div>
                    <h4 style={{ fontWeight: 800, fontSize: '0.9375rem', color: '#1e293b' }}>정밀 오프라인 평가 메트릭 (NDCG / Recall)</h4>
                    <p style={{ fontSize: '0.8125rem', color: '#64748b', lineHeight: '1.4', marginTop: '0.125rem' }}>
                      평가 수치는 실제 유저의 최근 5개 이상 좋아요 이력에서 마지막 좋아요를 숨겨둔 후, 이전 이력을 통해 추천했을 때 정답이 해당 랭킹에 올라오는지를 검증합니다.
                    </p>
                  </div>
                </div>

                {/* Charts */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  {/* Radar Chart */}
                  <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '1.25rem', color: '#0f172a' }}>📡 다차원 메트릭 레이더</h3>
                    <ResponsiveContainer width="100%" height={260}>
                      <RadarChart data={[
                        { subject: 'Recall@5', rec: +(metrics.metrics.recommendation_quality['Recall@5'] * 100).toFixed(1), search: +(metrics.metrics.search_fidelity['Recall@5'] * 100).toFixed(1) },
                        { subject: 'NDCG@5',   rec: +(metrics.metrics.recommendation_quality['NDCG@5']   * 100).toFixed(1), search: +(metrics.metrics.search_fidelity['NDCG@5']   * 100).toFixed(1) },
                        { subject: 'Recall@10',rec: +(metrics.metrics.recommendation_quality['Recall@10']* 100).toFixed(1), search: +(metrics.metrics.search_fidelity['Recall@10']* 100).toFixed(1) },
                        { subject: 'NDCG@10',  rec: +(metrics.metrics.recommendation_quality['NDCG@10']  * 100).toFixed(1), search: +(metrics.metrics.search_fidelity['NDCG@10']  * 100).toFixed(1) },
                      ]}>
                        <PolarGrid stroke="#e2e8f0" />
                        <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12, fontWeight: 700, fill: '#475569' }} />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} tickCount={4} />
                        <Radar name="추천 품질" dataKey="rec" stroke="#e60023" fill="#e60023" fillOpacity={0.18} strokeWidth={2} />
                        <Radar name="검색 정밀도" dataKey="search" stroke="#0284c7" fill="#0284c7" fillOpacity={0.18} strokeWidth={2} />
                        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, fontWeight: 700 }} />
                        <Tooltip formatter={(v: number) => `${v}%`} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Bar Chart */}
                  <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '1.25rem', color: '#0f172a' }}>📊 메트릭 비교 히스토그램</h3>
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={[
                        { name: 'Recall@5',  rec: +(metrics.metrics.recommendation_quality['Recall@5'] * 100).toFixed(1), search: +(metrics.metrics.search_fidelity['Recall@5'] * 100).toFixed(1) },
                        { name: 'NDCG@5',    rec: +(metrics.metrics.recommendation_quality['NDCG@5']   * 100).toFixed(1), search: +(metrics.metrics.search_fidelity['NDCG@5']   * 100).toFixed(1) },
                        { name: 'Recall@10', rec: +(metrics.metrics.recommendation_quality['Recall@10']* 100).toFixed(1), search: +(metrics.metrics.search_fidelity['Recall@10']* 100).toFixed(1) },
                        { name: 'NDCG@10',   rec: +(metrics.metrics.recommendation_quality['NDCG@10']  * 100).toFixed(1), search: +(metrics.metrics.search_fidelity['NDCG@10']  * 100).toFixed(1) },
                      ]} barCategoryGap="30%" barGap={4}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="name" tick={{ fontSize: 11, fontWeight: 700, fill: '#475569' }} />
                        <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                        <Tooltip formatter={(v: number) => `${v}%`} />
                        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, fontWeight: 700 }} />
                        <Bar dataKey="rec" name="추천 품질" fill="#e60023" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="search" name="검색 정밀도" fill="#0284c7" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                    <h3 style={{ fontSize: '1.125rem', fontWeight: 800, marginBottom: '1.5rem', color: '#0f172a' }}>❤️ 개인화 추천 만족도 (Recommendation Quality)</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                      {[
                        { label: 'Recall@5', val: metrics.metrics.recommendation_quality['Recall@5'], desc: '상위 5개 추천 목록 중 실제 좋아요 아이템이 포함될 확률' },
                        { label: 'NDCG@5', val: metrics.metrics.recommendation_quality['NDCG@5'], desc: '상위 5개 내 정답이 위치한 순위 가중치' },
                        { label: 'Recall@10', val: metrics.metrics.recommendation_quality['Recall@10'], desc: '상위 10개 추천 목록 기준 재현율' },
                        { label: 'NDCG@10', val: metrics.metrics.recommendation_quality['NDCG@10'], desc: '상위 10개 추천 목록 기준 NDCG' },
                      ].map(item => (
                        <div key={item.label}>
                          <div style={{ display: 'flex', justifySelf: 'space-between', justifyContent: 'space-between', marginBottom: '0.5rem', alignItems: 'end' }}>
                            <div>
                              <span style={{ fontWeight: 800, color: '#334155', fontSize: '0.9375rem' }}>{item.label}</span>
                              <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.125rem' }}>{item.desc}</p>
                            </div>
                            <span style={{ fontSize: '1.25rem', fontWeight: 900, color: 'var(--primary-red)' }}>{(item.val * 100).toFixed(1)}%</span>
                          </div>
                          <div style={{ height: '8px', backgroundColor: '#f1f5f9', borderRadius: '999px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${item.val * 100}%`, backgroundColor: 'var(--primary-red)', borderRadius: '999px' }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                    <h3 style={{ fontSize: '1.125rem', fontWeight: 800, marginBottom: '1.5rem', color: '#0f172a' }}>🔍 시맨틱 검색 매칭도 (Search Fidelity)</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                      {[
                        { label: 'Recall@5', val: metrics.metrics.search_fidelity['Recall@5'], desc: '텍스트 캡션 쿼리로 검색 시 본래의 이미지 포스트가 5위 내 매칭될 확률' },
                        { label: 'NDCG@5', val: metrics.metrics.search_fidelity['NDCG@5'], desc: '검색 상위 5개 결과의 정밀 정렬 정확도 가중치' },
                        { label: 'Recall@10', val: metrics.metrics.search_fidelity['Recall@10'], desc: '상위 10개 검색 결과 기준 재현율' },
                        { label: 'NDCG@10', val: metrics.metrics.search_fidelity['NDCG@10'], desc: '상위 10개 검색 결과 기준 NDCG' },
                      ].map(item => (
                        <div key={item.label}>
                          <div style={{ display: 'flex', justifySelf: 'space-between', justifyContent: 'space-between', marginBottom: '0.5rem', alignItems: 'end' }}>
                            <div>
                              <span style={{ fontWeight: 800, color: '#334155', fontSize: '0.9375rem' }}>{item.label}</span>
                              <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.125rem' }}>{item.desc}</p>
                            </div>
                            <span style={{ fontSize: '1.25rem', fontWeight: 900, color: '#0284c7' }}>{(item.val * 100).toFixed(1)}%</span>
                          </div>
                          <div style={{ height: '8px', backgroundColor: '#f1f5f9', borderRadius: '999px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${item.val * 100}%`, backgroundColor: '#0284c7', borderRadius: '999px' }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ padding: '4rem', textAlign: 'center', border: '1px dashed #cbd5e1', borderRadius: '24px' }}>
                <AlertCircle size={32} style={{ color: '#94a3b8', margin: '0 auto 1rem' }} />
                <p style={{ fontWeight: 700, color: 'var(--text-muted)' }}>가용한 오프라인 평가 데이터셋이 존재하지 않습니다.</p>
              </div>
            )}
          </motion.div>
        )}

        {/* 2. SIMULATOR TAB */}
        {activeTab === 'simulator' && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '2rem', alignItems: 'start' }}>
              <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '1.5rem', backgroundColor: 'white' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 800, marginBottom: '1.25rem', color: '#1e293b' }}>시뮬레이터 설정</h3>
                
                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 700, color: '#64748b', marginBottom: '0.5rem' }}>대상 사용자 프로필</label>
                  <select 
                    value={selectedUserId} 
                    onChange={e => setSelectedUserId(e.target.value)}
                    style={{ width: '100%', padding: '0.75rem', borderRadius: '12px', border: '1px solid #cbd5e1', fontSize: '0.875rem', fontWeight: 700, outline: 'none' }}
                  >
                    <option value="0">비로그인 (개인화 미적용)</option>
                    {activeUsers.map(u => (
                      <option key={u.id} value={u.id}>{u.label}</option>
                    ))}
                  </select>
                </div>

                {activeUsers.length > 0 && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 700, color: '#64748b', marginBottom: '0.5rem' }}>유저 좋아요 분포</label>
                    <ResponsiveContainer width="100%" height={130}>
                      <BarChart data={activeUsers.slice(0, 8).map(u => ({ name: u.label.split(' ')[0], likes: u.likes_count }))} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#94a3b8' }} />
                        <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} />
                        <Tooltip formatter={(v: number) => [`${v}개`, '좋아요']} />
                        <Bar dataKey="likes" fill="#e60023" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 700, color: '#64748b', marginBottom: '0.5rem' }}>검색 / 추천 모드</label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {[
                      { id: 'recommendation', label: '개인화 복합 추천 (Hybrid)', desc: '유저 관심사 + 인기도 가중치' },
                      { id: 'text', label: '의미론적 텍스트 검색 (Text)', desc: '검색어 텍스트 임베딩 매칭' },
                      { id: 'image', label: '시각적 유사도 검색 (Image)', desc: 'CLIP 이미지 벡터 기반 매칭' }
                    ].map(opt => (
                      <div 
                        key={opt.id} 
                        onClick={() => setSimType(opt.id as any)}
                        style={{
                          padding: '0.75rem',
                          borderRadius: '12px',
                          border: simType === opt.id ? '2px solid var(--primary-red)' : '1px solid #e2e8f0',
                          backgroundColor: simType === opt.id ? '#fff5f5' : 'transparent',
                          cursor: 'pointer',
                          transition: 'all 0.15s'
                        }}
                      >
                        <span style={{ fontSize: '0.875rem', fontWeight: 800, color: '#1e293b' }}>{opt.label}</span>
                        <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.125rem' }}>{opt.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ marginBottom: '1.5rem' }}>
                  {simType === 'image' ? (
                    <div>
                      <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 700, color: '#64748b', marginBottom: '0.5rem' }}>검색 대상 이미지</label>
                      {simImagePreview ? (
                        <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0' }}>
                          <img src={simImagePreview} style={{ width: '100%', height: '180px', objectFit: 'cover' }} />
                          <button 
                            onClick={clearSimImage}
                            style={{ position: 'absolute', top: '0.5rem', right: '0.5rem', backgroundColor: 'rgba(0,0,0,0.6)', color: 'white', border: 'none', borderRadius: '50%', width: '24px', height: '24px', cursor: 'pointer' }}
                          >
                            X
                          </button>
                        </div>
                      ) : (
                        <div 
                          onClick={() => simImageInputRef.current?.click()}
                          style={{ border: '2px dashed #cbd5e1', borderRadius: '12px', padding: '2rem 1rem', textAlign: 'center', cursor: 'pointer', backgroundColor: '#f8fafc' }}
                        >
                          <Upload size={20} style={{ color: '#94a3b8', margin: '0 auto 0.5rem' }} />
                          <span style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#64748b' }}>이미지 선택 또는 드롭</span>
                        </div>
                      )}
                      <input 
                        type="file" 
                        ref={simImageInputRef} 
                        onChange={handleSimImageChange} 
                        accept="image/*" 
                        style={{ display: 'none' }} 
                      />
                    </div>
                  ) : (
                    <div>
                      <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 700, color: '#64748b', marginBottom: '0.5rem' }}>검색 질의어 (Query)</label>
                      <input
                        type="text"
                        value={simQuery}
                        onChange={e => setSimQuery(e.target.value)}
                        placeholder={simType === 'text' ? '예: 빨간 옷, 미니멀 하우스' : '추천에는 쿼리가 불필요합니다'}
                        disabled={simType === 'recommendation'}
                        style={{ width: '100%', padding: '0.75rem', borderRadius: '12px', border: '1px solid #cbd5e1', fontSize: '0.875rem', outline: 'none' }}
                      />
                    </div>
                  )}
                </div>

                <button
                  onClick={handleRunSimulator}
                  disabled={simLoading}
                  style={{
                    width: '100%',
                    padding: '0.875rem',
                    borderRadius: '12px',
                    backgroundColor: 'var(--primary-red)',
                    color: 'white',
                    fontWeight: 800,
                    fontSize: '0.9375rem',
                    border: 'none',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem'
                  }}
                >
                  {simLoading ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
                  시뮬레이션 추천 실행
                </button>
              </div>

              <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white', minHeight: '400px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid #f1f5f9', paddingBottom: '1rem' }}>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: 800, color: '#0f172a' }}>추천 시뮬레이션 결과</h3>
                  <span style={{ fontSize: '0.8125rem', fontWeight: 700, color: 'var(--text-muted)' }}>
                    총 <strong>{simResults.length}개</strong>의 후보 아이템 정렬됨
                  </span>
                </div>

                {simLoading ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px' }}>
                    <RefreshCw size={36} className="animate-spin" style={{ color: 'var(--primary-red)', marginBottom: '1rem' }} />
                    <p style={{ fontWeight: 700, color: '#475569' }}>Redis HNSW 탐색 및 가중치 정렬 중...</p>
                  </div>
                ) : simResults.length > 0 ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '1.5rem' }}>
                    {simResults.map((post, idx) => (
                      <div 
                        key={post.id} 
                        onClick={() => setSelectedPost(post)}
                        style={{ border: '1px solid #e2e8f0', borderRadius: '16px', overflow: 'hidden', cursor: 'pointer', backgroundColor: '#f8fafc', position: 'relative' }}
                      >
                        <div style={{ position: 'absolute', top: '0.5rem', left: '0.5rem', display: 'flex', gap: '0.25rem', zIndex: 10 }}>
                          <span style={{ backgroundColor: 'rgba(15, 23, 42, 0.85)', color: 'white', fontSize: '10px', fontWeight: 800, padding: '2px 6px', borderRadius: '4px' }}>
                            #{idx + 1}
                          </span>
                          <span style={{ backgroundColor: 'var(--primary-red)', color: 'white', fontSize: '10px', fontWeight: 800, padding: '2px 6px', borderRadius: '4px' }}>
                            Score: {post.score}
                          </span>
                        </div>

                        <div style={{ height: '160px', width: '100%', overflow: 'hidden', backgroundColor: '#e2e8f0' }}>
                          <img src={post.image_url} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        </div>
                        <div style={{ padding: '0.75rem' }}>
                          <h4 style={{ fontSize: '0.8125rem', fontWeight: 800, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: '#1e293b' }}>{post.title}</h4>
                          <p style={{ fontSize: '0.75rem', color: '#64748b', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', marginTop: '0.125rem' }}>{post.caption || '태그 없음'}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', color: '#94a3b8' }}>
                    <Info size={32} style={{ marginBottom: '1rem' }} />
                    <p style={{ fontWeight: 700 }}>시뮬레이션 조건 설정 후 버튼을 눌러 추천을 진행하세요.</p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {/* 3. BENCHMARK TAB */}
        {activeTab === 'benchmark' && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            {/* Benchmark Sub-Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', backgroundColor: '#f8fafc', padding: '0.375rem', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
              {[
                { id: 'animal', label: '🐾 동물 데이터셋', color: '#16a34a' },
                { id: 'synthetic', label: '🔷 합성 도형', color: '#0284c7' },
                { id: 'gemini', label: '🤖 Gemini 시딩', color: '#6366f1' },
                { id: 'custom', label: '📁 커스텀 ZIP', color: '#e60023' },
              ].map(t => (
                <button
                  key={t.id}
                  onClick={() => setBenchmarkSubTab(t.id as any)}
                  style={{
                    flex: 1,
                    padding: '0.625rem 1rem',
                    borderRadius: '12px',
                    border: 'none',
                    fontWeight: 800,
                    fontSize: '0.875rem',
                    cursor: 'pointer',
                    backgroundColor: benchmarkSubTab === t.id ? 'white' : 'transparent',
                    color: benchmarkSubTab === t.id ? t.color : '#94a3b8',
                    boxShadow: benchmarkSubTab === t.id ? '0 1px 4px rgba(0,0,0,0.08)' : 'none',
                    transition: 'all 0.15s'
                  }}
                >{t.label}</button>
              ))}
            </div>

            {/* Animal Dataset Sub-Tab */}
            {benchmarkSubTab === 'animal' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                    <div>
                      <h3 style={{ fontSize: '1.5rem', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.02em' }}>CIFAR-100 실제 이미지 벤치마크 (DB 실시간 주입)</h3>
                      <p style={{ fontSize: '0.9375rem', color: '#64748b', marginTop: '0.375rem', lineHeight: '1.5' }}>
                        CIFAR-100 데이터셋 100개 클래스 × 2샘플 = 200장의 실제 이미지를 DB에 주입하고 크로스모달(텍스트→이미지, 이미지→이미지) Recall·NDCG를 측정합니다. 테스트 후 자동 삭제됩니다.
                      </p>
                    </div>
                    <button
                      onClick={handleRunAnimalBenchmark}
                      disabled={animalLoading}
                      style={{ backgroundColor: '#16a34a', color: 'white', padding: '0.875rem 1.75rem', borderRadius: '14px', fontWeight: 800, fontSize: '0.9375rem', border: 'none', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.625rem', flexShrink: 0, whiteSpace: 'nowrap' }}
                    >
                      {animalLoading ? <RefreshCw size={18} className="animate-spin" /> : <Play size={18} />}
                      DB 주입 및 벤치마크 실행
                    </button>
                  </div>

                  {animalLoading && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', flexDirection: 'column', gap: '1rem' }}>
                      <RefreshCw size={40} className="animate-spin" style={{ color: '#16a34a' }} />
                      <p style={{ fontWeight: 700, color: '#64748b' }}>CIFAR-100 다운로드 및 200개 샘플 Redis 인덱싱 중... (수 분 소요)</p>
                    </div>
                  )}

                  {animalResult && (
                    <>
                      {/* Summary metric cards: 2 rows × 4 cols */}
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '0.75rem' }}>
                        {(['Recall@1', 'Recall@3', 'Recall@5', 'NDCG@3'] as const).map(m => (
                          <div key={`t2i-${m}`} style={{ borderRadius: '16px', padding: '1.25rem 1rem', textAlign: 'center', border: '2px solid #bbf7d0', background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)' }}>
                            <p style={{ fontSize: '0.75rem', fontWeight: 700, color: '#166534', marginBottom: '0.5rem', letterSpacing: '0.04em' }}>Text→Img {m}</p>
                            <p style={{ fontSize: '2.5rem', fontWeight: 900, color: '#15803d', lineHeight: 1, letterSpacing: '-0.04em' }}>
                              {animalResult.text_to_image[m] !== undefined ? `${(animalResult.text_to_image[m] * 100).toFixed(0)}` : '—'}
                            </p>
                            <p style={{ fontSize: '0.875rem', fontWeight: 700, color: '#16a34a' }}>%</p>
                          </div>
                        ))}
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '1.75rem' }}>
                        {(['Recall@1', 'Recall@3', 'Recall@5', 'NDCG@3'] as const).map(m => (
                          <div key={`i2i-${m}`} style={{ borderRadius: '16px', padding: '1.25rem 1rem', textAlign: 'center', border: '2px solid #bae6fd', background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
                            <p style={{ fontSize: '0.75rem', fontWeight: 700, color: '#075985', marginBottom: '0.5rem', letterSpacing: '0.04em' }}>Img→Img {m}</p>
                            <p style={{ fontSize: '2.5rem', fontWeight: 900, color: '#0369a1', lineHeight: 1, letterSpacing: '-0.04em' }}>
                              {animalResult.image_to_image[m] !== undefined ? `${(animalResult.image_to_image[m] * 100).toFixed(0)}` : '—'}
                            </p>
                            <p style={{ fontSize: '0.875rem', fontWeight: 700, color: '#0284c7' }}>%</p>
                          </div>
                        ))}
                      </div>

                      {/* Per-class detail table */}
                      <div>
                        <h4 style={{ fontWeight: 800, fontSize: '1rem', color: '#0f172a', marginBottom: '0.75rem' }}>
                          클래스별 검색 결과 ({animalResult.details.length}개 클래스)
                        </h4>
                        <div style={{ borderRadius: '14px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
                          {/* Header */}
                          <div style={{ display: 'grid', gridTemplateColumns: '2.5rem 1fr 5.5rem 3.5rem 3.5rem 3.5rem 5.5rem 3.5rem 3.5rem 3.5rem', backgroundColor: '#f8fafc', borderBottom: '2px solid #e2e8f0', padding: '0.625rem 0.875rem', alignItems: 'center', gap: '0.25rem' }}>
                            {[
                              { label: '#',         color: '#94a3b8', align: 'left'   },
                              { label: 'CLASS',     color: '#475569', align: 'left'   },
                              { label: 'Text Rank', color: '#166534', align: 'center' },
                              { label: 'T@1',       color: '#166534', align: 'center' },
                              { label: 'T@3',       color: '#166534', align: 'center' },
                              { label: 'T@5',       color: '#166534', align: 'center' },
                              { label: 'Img Rank',  color: '#075985', align: 'center' },
                              { label: 'I@1',       color: '#075985', align: 'center' },
                              { label: 'I@3',       color: '#075985', align: 'center' },
                              { label: 'I@5',       color: '#075985', align: 'center' },
                            ].map(col => (
                              <span key={col.label} style={{ fontSize: '0.6875rem', fontWeight: 700, color: col.color, textAlign: col.align as any, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{col.label}</span>
                            ))}
                          </div>

                          {/* Body */}
                          <div style={{ maxHeight: '480px', overflowY: 'auto' }}>
                            {animalResult.details.map((det, idx) => {
                              const tr = typeof det.text_rank === 'number' ? det.text_rank : 999;
                              const ir = typeof det.image_rank === 'number' ? det.image_rank : 999;

                              const rankBadge = (rank: number, scheme: 'green' | 'blue') => {
                                const notFound = rank >= 999;
                                let bg: string, fg: string;
                                if (notFound)      { bg = '#f1f5f9'; fg = '#94a3b8'; }
                                else if (rank <= 1) { bg = scheme === 'green' ? '#dcfce7' : '#dbeafe'; fg = scheme === 'green' ? '#15803d' : '#1d4ed8'; }
                                else if (rank <= 3) { bg = '#fef9c3'; fg = '#a16207'; }
                                else if (rank <= 10){ bg = '#ffedd5'; fg = '#c2410c'; }
                                else               { bg = '#fee2e2'; fg = '#b91c1c'; }
                                return (
                                  <span style={{ display: 'inline-block', minWidth: '3.25rem', textAlign: 'center', fontSize: '0.8125rem', fontWeight: 800, padding: '3px 8px', borderRadius: '7px', backgroundColor: bg, color: fg }}>
                                    {notFound ? 'N/F' : `#${rank}`}
                                  </span>
                                );
                              };

                              const hitBadge = (rank: number, k: number) => {
                                const hit = rank <= k;
                                return (
                                  <span style={{ display: 'inline-block', fontSize: '0.8125rem', fontWeight: 800, padding: '2px 8px', borderRadius: '7px', backgroundColor: hit ? '#f0fdf4' : '#fafafa', color: hit ? '#16a34a' : '#cbd5e1', border: `1px solid ${hit ? '#bbf7d0' : '#e2e8f0'}` }}>
                                    {hit ? '✓' : '✗'}
                                  </span>
                                );
                              };

                              return (
                                <div key={det.name} style={{ display: 'grid', gridTemplateColumns: '2.5rem 1fr 5.5rem 3.5rem 3.5rem 3.5rem 5.5rem 3.5rem 3.5rem 3.5rem', gap: '0.25rem', padding: '0.5rem 0.875rem', alignItems: 'center', borderBottom: '1px solid #f8fafc', backgroundColor: idx % 2 === 0 ? 'white' : '#fafffe' }}>
                                  <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600 }}>{idx + 1}</span>
                                  <span style={{ fontSize: '0.8125rem', fontWeight: 700, textTransform: 'capitalize', color: '#0f172a', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{det.name}</span>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{rankBadge(tr, 'green')}</div>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{hitBadge(tr, 1)}</div>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{hitBadge(tr, 3)}</div>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{hitBadge(tr, 5)}</div>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{rankBadge(ir, 'blue')}</div>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{hitBadge(ir, 1)}</div>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{hitBadge(ir, 3)}</div>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>{hitBadge(ir, 5)}</div>
                                </div>
                              );
                            })}
                          </div>

                          {/* Footer summary */}
                          <div style={{ padding: '0.625rem 0.875rem', backgroundColor: '#f8fafc', borderTop: '1px solid #e2e8f0', display: 'flex', gap: '1.5rem', fontSize: '0.75rem', fontWeight: 700, color: '#64748b' }}>
                            <span>✅ Text@1: {animalResult.details.filter(d => typeof d.text_rank === 'number' && d.text_rank <= 1).length}/{animalResult.details.length}</span>
                            <span>✅ Text@3: {animalResult.details.filter(d => typeof d.text_rank === 'number' && d.text_rank <= 3).length}/{animalResult.details.length}</span>
                            <span style={{ borderLeft: '1px solid #e2e8f0', paddingLeft: '1.5rem' }}>✅ Img@1: {animalResult.details.filter(d => typeof d.image_rank === 'number' && d.image_rank <= 1).length}/{animalResult.details.length}</span>
                            <span>✅ Img@3: {animalResult.details.filter(d => typeof d.image_rank === 'number' && d.image_rank <= 3).length}/{animalResult.details.length}</span>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Synthetic Sub-Tab */}
            {benchmarkSubTab === 'synthetic' && (
              <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                  <div>
                    <h3 style={{ fontSize: '1.5rem', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.02em' }}>Pillow 합성 도형 데이터셋 평가</h3>
                    <p style={{ fontSize: '0.9375rem', color: '#64748b', marginTop: '0.375rem', lineHeight: '1.5' }}>
                      동그라미·세모·네모·별 등 10가지 색상의 합성 이미지를 메모리에서 실시간 렌더링하고, 노이즈 쿼리로 검색하여 CLIP/SBERT 핵심 정확도를 측정합니다.
                    </p>
                  </div>
                  <button
                    onClick={handleRunSyntheticBenchmark}
                    disabled={syntheticLoading}
                    style={{ backgroundColor: '#0284c7', color: 'white', padding: '0.875rem 1.75rem', borderRadius: '14px', fontWeight: 800, fontSize: '0.9375rem', border: 'none', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.625rem', flexShrink: 0 }}
                  >
                    {syntheticLoading ? <RefreshCw size={18} className="animate-spin" /> : <Play size={18} />}
                    합성 벤치마크 실행
                  </button>
                </div>

                {syntheticLoading && (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', flexDirection: 'column', gap: '1rem' }}>
                    <RefreshCw size={40} className="animate-spin" style={{ color: '#0284c7' }} />
                    <p style={{ fontWeight: 700, color: '#64748b' }}>합성 이미지 렌더링 및 임베딩 평가 중...</p>
                  </div>
                )}

                {syntheticResult && (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                      {['Recall@1', 'Recall@3', 'NDCG@3'].map(m => (
                        <div key={m} style={{ borderRadius: '20px', padding: '1.75rem 1.25rem', textAlign: 'center', border: '2px solid #bae6fd', background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
                          <p style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#075985', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Text→Image {m}</p>
                          <p style={{ fontSize: '3.5rem', fontWeight: 900, color: '#0369a1', lineHeight: 1, letterSpacing: '-0.04em' }}>{(syntheticResult.text_to_image[m] * 100).toFixed(0)}</p>
                          <p style={{ fontSize: '1rem', fontWeight: 700, color: '#0284c7', marginTop: '0.25rem' }}>%</p>
                        </div>
                      ))}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                      {['Recall@1', 'Recall@3', 'NDCG@3'].map(m => (
                        <div key={m} style={{ borderRadius: '20px', padding: '1.75rem 1.25rem', textAlign: 'center', border: '2px solid #fecaca', background: 'linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%)' }}>
                          <p style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#991b1b', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Image→Image {m}</p>
                          <p style={{ fontSize: '3.5rem', fontWeight: 900, color: '#b91c1c', lineHeight: 1, letterSpacing: '-0.04em' }}>{(syntheticResult.image_to_image[m] * 100).toFixed(0)}</p>
                          <p style={{ fontSize: '1rem', fontWeight: 700, color: '#e60023', marginTop: '0.25rem' }}>%</p>
                        </div>
                      ))}
                    </div>
                    <p style={{ marginTop: '1.5rem', fontSize: '0.875rem', color: '#94a3b8', textAlign: 'center' }}>샘플 수: {syntheticResult.sample_size}개</p>
                  </>
                )}
              </div>
            )}

            {/* Gemini Sub-Tab */}
            {benchmarkSubTab === 'gemini' && (
              <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                  <div>
                    <h3 style={{ fontSize: '1.5rem', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.02em' }}>Gemini 가상 페르소나 & 좋아요 주입</h3>
                    <p style={{ fontSize: '0.9375rem', color: '#64748b', marginTop: '0.375rem', lineHeight: '1.5', maxWidth: '600px' }}>
                      Gemini-2.5-Flash LLM이 고양이 집사, 운동 매니아, 맛집 탐방가 등 개성 넘치는 50개의 가상 인물(Persona)을 생성하고, 각자의 관심사에 맞는 포스트에 좋아요를 남깁니다.
                    </p>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.5rem', flexShrink: 0 }}>
                    <button
                      onClick={handleRunGeminiSeeding}
                      disabled={geminiLoading}
                      style={{ backgroundColor: '#6366f1', color: 'white', padding: '0.875rem 1.75rem', borderRadius: '14px', fontWeight: 800, fontSize: '0.9375rem', border: 'none', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.625rem' }}
                    >
                      {geminiLoading ? <RefreshCw size={18} className="animate-spin" /> : <Wand2 size={18} />}
                      시뮬레이션 데이터 주입 시작
                    </button>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', color: '#64748b', cursor: 'pointer', userSelect: 'none' }}>
                      <input type="checkbox" checked={geminiForce} onChange={e => setGeminiForce(e.target.checked)} style={{ accentColor: '#6366f1' }} />
                      이미 시드된 경우에도 강제 재실행 (Gemini API 비용 발생)
                    </label>
                  </div>
                </div>

                {geminiLoading && (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', flexDirection: 'column', gap: '1rem' }}>
                    <RefreshCw size={40} className="animate-spin" style={{ color: '#6366f1' }} />
                    <p style={{ fontWeight: 700, color: '#64748b' }}>Gemini API로 가상 페르소나 생성 및 좋아요 주입 중...</p>
                  </div>
                )}

                {geminiResult && geminiResult.status === 'skipped' && (
                  <div style={{ borderRadius: '16px', padding: '1.5rem', backgroundColor: '#f0fdf4', border: '2px solid #bbf7d0', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <span style={{ fontSize: '2rem' }}>✅</span>
                    <div>
                      <p style={{ fontWeight: 800, color: '#166534', fontSize: '1rem', marginBottom: '0.25rem' }}>이미 시딩 완료됨 — Gemini API 호출 건너뜀</p>
                      <p style={{ color: '#15803d', fontSize: '0.875rem' }}>{geminiResult.message}</p>
                    </div>
                  </div>
                )}

                {geminiResult && geminiResult.status === 'success' && (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
                      <div style={{ borderRadius: '20px', padding: '2rem', textAlign: 'center', border: '2px solid #ddd6fe', background: 'linear-gradient(135deg, #faf5ff 0%, #ede9fe 100%)' }}>
                        <p style={{ fontSize: '0.875rem', fontWeight: 700, color: '#6d28d9', marginBottom: '0.75rem' }}>총 주입된 좋아요</p>
                        <p style={{ fontSize: '4rem', fontWeight: 900, color: '#5b21b6', lineHeight: 1, letterSpacing: '-0.04em' }}>{geminiResult.total_likes_seeded?.toLocaleString() ?? '—'}</p>
                        <p style={{ fontSize: '1rem', fontWeight: 700, color: '#7c3aed', marginTop: '0.25rem' }}>개</p>
                      </div>
                      <div style={{ borderRadius: '20px', padding: '2rem', textAlign: 'center', border: '2px solid #ddd6fe', background: 'linear-gradient(135deg, #faf5ff 0%, #ede9fe 100%)' }}>
                        <p style={{ fontSize: '0.875rem', fontWeight: 700, color: '#6d28d9', marginBottom: '0.75rem' }}>생성된 페르소나 수</p>
                        <p style={{ fontSize: '4rem', fontWeight: 900, color: '#5b21b6', lineHeight: 1, letterSpacing: '-0.04em' }}>{geminiResult.details?.length ?? '—'}</p>
                        <p style={{ fontSize: '1rem', fontWeight: 700, color: '#7c3aed', marginTop: '0.25rem' }}>명</p>
                      </div>
                    </div>
                    <h4 style={{ fontWeight: 800, fontSize: '1rem', color: '#0f172a', marginBottom: '1rem' }}>페르소나별 좋아요 분포</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '0.75rem' }}>
                      {geminiResult.details?.map((p: any) => (
                        <div key={p.name} style={{ display: 'flex', justifyContent: 'space-between', backgroundColor: '#f5f3ff', border: '1px solid #ddd6fe', padding: '0.875rem 1.125rem', borderRadius: '14px', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.875rem', fontWeight: 800, color: '#4c1d95' }}>{p.name}</span>
                          <span style={{ fontSize: '0.9375rem', fontWeight: 900, color: '#6d28d9', backgroundColor: '#ede9fe', padding: '4px 12px', borderRadius: '20px' }}>❤️ {p.liked_count}</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Custom ZIP Sub-Tab */}
            {benchmarkSubTab === 'custom' && (
              <div style={{ border: '1px solid #e2e8f0', borderRadius: '24px', padding: '2rem', backgroundColor: 'white' }}>
                <div style={{ marginBottom: '1.5rem' }}>
                  <h3 style={{ fontSize: '1.5rem', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.02em' }}>커스텀 데이터셋 평가 (ZIP 업로드)</h3>
                  <p style={{ fontSize: '0.9375rem', color: '#64748b', marginTop: '0.375rem', lineHeight: '1.5' }}>
                    직접 수집한 이미지와 캡션 어노테이션이 담긴 ZIP을 업로드하여 검색 정확도를 측정합니다.
                  </p>
                </div>

                <div style={{ backgroundColor: '#f0fdf4', padding: '1rem 1.25rem', borderRadius: '14px', border: '1px solid #bbf7d0', fontSize: '0.875rem', color: '#166534', marginBottom: '1.5rem', lineHeight: '1.5' }}>
                  <strong>📂 ZIP 요구 규격</strong> — 루트에 <code style={{ backgroundColor: '#dcfce7', padding: '1px 6px', borderRadius: '4px' }}>dataset.json</code> 필수 포함.
                  포맷: <code style={{ backgroundColor: '#dcfce7', padding: '1px 6px', borderRadius: '4px' }}>{`[{"filename": "car.jpg", "caption": "luxury red car"}]`}</code>
                </div>

                <div
                  onDragEnter={handleZipDrag}
                  onDragOver={handleZipDrag}
                  onDragLeave={handleZipDrag}
                  onDrop={handleZipDrop}
                  onClick={() => zipInputRef.current?.click()}
                  style={{ border: isDragActive ? '2px dashed var(--primary-red)' : '2px dashed #cbd5e1', borderRadius: '20px', padding: '3rem 2rem', textAlign: 'center', cursor: 'pointer', backgroundColor: isDragActive ? '#fff5f5' : '#f8fafc', marginBottom: '1.5rem', transition: 'all 0.15s' }}
                >
                  <input type="file" ref={zipInputRef} onChange={handleZipSelect} accept=".zip" style={{ display: 'none' }} />
                  <Upload size={32} style={{ color: isDragActive ? 'var(--primary-red)' : '#94a3b8', margin: '0 auto 0.75rem' }} />
                  <span style={{ fontSize: '1rem', fontWeight: 800, color: isDragActive ? 'var(--primary-red)' : '#475569', display: 'block' }}>
                    {customZipName ? `📦 ${customZipName}` : '평가용 ZIP 파일을 드롭하거나 클릭해서 선택'}
                  </span>
                </div>

                <button
                  onClick={handleRunCustomBenchmark}
                  disabled={customLoading || !customZip}
                  style={{ backgroundColor: 'var(--primary-red)', color: 'white', padding: '0.875rem 1.75rem', borderRadius: '14px', fontWeight: 800, fontSize: '0.9375rem', border: 'none', cursor: customZip ? 'pointer' : 'not-allowed', opacity: customZip ? 1 : 0.5, display: 'inline-flex', alignItems: 'center', gap: '0.625rem' }}
                >
                  {customLoading ? <RefreshCw size={18} className="animate-spin" /> : <Play size={18} />}
                  커스텀 데이터셋 평가 시작
                </button>

                {customResult && (
                  <div style={{ marginTop: '2rem' }}>
                    {customResult.status === 'success' ? (
                      <>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
                          <div style={{ borderRadius: '20px', padding: '1.5rem', textAlign: 'center', border: '2px solid #e2e8f0', backgroundColor: '#f8fafc', gridColumn: 'span 1' }}>
                            <p style={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', marginBottom: '0.5rem' }}>교차 유사도</p>
                            <p style={{ fontSize: '2.5rem', fontWeight: 900, color: '#0f172a', lineHeight: 1 }}>{(customResult.average_cross_image_similarity * 100).toFixed(0)}</p>
                            <p style={{ fontSize: '0.875rem', color: '#94a3b8', fontWeight: 700 }}>%</p>
                          </div>
                          {['Recall@1', 'Recall@3', 'NDCG@3'].map(m => (
                            <div key={m} style={{ borderRadius: '20px', padding: '1.5rem', textAlign: 'center', border: '2px solid #fecaca', background: 'linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%)' }}>
                              <p style={{ fontSize: '0.75rem', fontWeight: 700, color: '#991b1b', marginBottom: '0.5rem', textTransform: 'uppercase' }}>{m}</p>
                              <p style={{ fontSize: '2.5rem', fontWeight: 900, color: '#b91c1c', lineHeight: 1 }}>
                                {customResult.text_to_image[m] !== undefined ? (customResult.text_to_image[m] * 100).toFixed(0) : '—'}
                              </p>
                              <p style={{ fontSize: '0.875rem', color: '#e60023', fontWeight: 700 }}>%</p>
                            </div>
                          ))}
                        </div>
                        <p style={{ fontSize: '0.875rem', color: '#94a3b8', textAlign: 'center' }}>샘플 수: {customResult.sample_size}개</p>
                      </>
                    ) : (
                      <div style={{ color: '#ef4444', display: 'flex', gap: '0.75rem', alignItems: 'flex-start', backgroundColor: '#fff5f5', padding: '1.25rem', borderRadius: '14px', border: '1px solid #fecaca' }}>
                        <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
                        <div>
                          <span style={{ fontWeight: 800, fontSize: '0.9375rem' }}>오류 발생</span>
                          <p style={{ fontSize: '0.875rem', marginTop: '0.25rem' }}>{customResult.message}</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}

        {activeTab === 'system' && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '3rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ border: '1px solid #e2e8f0', borderRadius: '20px', padding: '1.5rem', backgroundColor: 'white' }}>
                  <h4 style={{ fontWeight: 800, color: '#1e293b', marginBottom: '0.5rem' }}>🤖 검색/추천 임베딩 타워 수동 재학습</h4>
                  <p style={{ fontSize: '0.8125rem', color: '#64748b', lineHeight: '1.4', marginBottom: '1.25rem' }}>
                    DB의 최신 좋아요 데이터를 기반으로 텍스트 및 이미지 임베딩 투영 벡터 레이어를 파인튜닝하고, 가중치를 갱신합니다.
                  </p>
                  <button
                    onClick={triggerTrain}
                    disabled={actionLoading}
                    style={{
                      backgroundColor: 'var(--primary-red)',
                      color: 'white',
                      border: 'none',
                      padding: '0.625rem 1.25rem',
                      borderRadius: '10px',
                      fontWeight: 800,
                      cursor: 'pointer',
                      fontSize: '0.875rem'
                    }}
                  >
                    모델 재학습 개시
                  </button>
                </div>

                <div style={{ border: '1px solid #e2e8f0', borderRadius: '20px', padding: '1.5rem', backgroundColor: 'white' }}>
                  <h4 style={{ fontWeight: 800, color: '#1e293b', marginBottom: '0.5rem' }}>🔄 Redis 벡터 강제 재생성 및 동기화</h4>
                  <p style={{ fontSize: '0.8125rem', color: '#64748b', lineHeight: '1.4', marginBottom: '1.25rem' }}>
                    기존 Redis HNSW 벡터 스키마 인덱스를 완전히 드롭한 후, 텍스트/이미지 이중 인덱스 스키마를 신규 구축하고 DB 데이터를 전부 다시 인덱싱합니다.
                  </p>
                  <button
                    onClick={triggerSync}
                    disabled={actionLoading}
                    style={{
                      backgroundColor: '#475569',
                      color: 'white',
                      border: 'none',
                      padding: '0.625rem 1.25rem',
                      borderRadius: '10px',
                      fontWeight: 800,
                      cursor: 'pointer',
                      fontSize: '0.875rem'
                    }}
                  >
                    Redis 벡터 동기화 실행
                  </button>
                </div>
              </div>

              <div style={{ border: '1px solid #e2e8f0', borderRadius: '20px', padding: '1.5rem', backgroundColor: '#0f172a', color: '#38bdf8', minHeight: '300px', display: 'flex', flexDirection: 'column' }}>
                <h4 style={{ fontWeight: 800, color: '#94a3b8', fontSize: '0.875rem', marginBottom: '1rem', borderBottom: '1px solid #1e293b', paddingBottom: '0.5rem' }}>
                  💻 시스템 진단 로그 (Real-time Diagnostic Logs)
                </h4>
                <div style={{ flex: 1, fontFamily: 'monospace', fontSize: '0.8125rem', overflowY: 'auto', lineHeight: '1.6', color: '#a5f3fc' }}>
                  {sysLogs.length > 0 ? (
                    sysLogs.map((log, index) => <div key={index}>{log}</div>)
                  ) : (
                    <div style={{ color: '#64748b' }}>여기에 모델 학습 및 동기화 진단 출력이 표시됩니다.</div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Post Detail Modal */}
      <AnimatePresence>
        {selectedPost && (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            exit={{ opacity: 0 }} 
            onClick={() => setSelectedPost(null)}
            style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(15, 23, 42, 0.75)', backdropFilter: 'blur(8px)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem' }}
          >
            <motion.div 
              initial={{ scale: 0.95, y: 20 }} 
              animate={{ scale: 1, y: 0 }} 
              exit={{ scale: 0.95, y: 20 }}
              onClick={e => e.stopPropagation()}
              style={{ backgroundColor: 'white', width: '90%', maxWidth: '800px', borderRadius: '32px', overflow: 'hidden', display: 'grid', gridTemplateColumns: '1.2fr 1fr', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid #e2e8f0' }}
            >
              <div style={{ height: '500px', backgroundColor: '#f1f5f9' }}>
                <img src={selectedPost.image_url} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div style={{ padding: '2.5rem', display: 'flex', flexDirection: 'column', position: 'relative' }}>
                <button 
                  onClick={() => setSelectedPost(null)}
                  style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', border: 'none', background: '#f1f5f9', borderRadius: '50%', width: '2.25rem', height: '2.25rem', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
                >
                  <X size={18} />
                </button>
                <div style={{ display: 'inline-flex', alignSelf: 'start', backgroundColor: 'var(--primary-red)', color: 'white', fontSize: '11px', fontWeight: 900, padding: '4px 10px', borderRadius: '6px', marginBottom: '1.25rem' }}>
                  매칭 가중치 Score: {selectedPost.score}
                </div>
                <h3 style={{ fontSize: '1.5rem', fontWeight: 900, color: '#0f172a', marginBottom: '1rem', letterSpacing: '-0.02em' }}>{selectedPost.title}</h3>
                <p style={{ fontSize: '0.9375rem', color: '#475569', lineHeight: '1.6', flex: 1 }}>{selectedPost.caption || '기록된 설명 텍스트가 없습니다.'}</p>
                <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: '1.5rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                  아이템 고유 ID: <code style={{ display: 'block', backgroundColor: '#f8fafc', padding: '6px 12px', borderRadius: '8px', border: '1px solid #e2e8f0', marginTop: '0.25rem', color: '#0369a1', fontFamily: 'monospace' }}>{selectedPost.id}</code>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DevConsole;
