import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from '@tanstack/react-query';
import client from '../api/client';

export const usePostStats = (postId: string, initialStats?: any) => {
  return useQuery({
    queryKey: ['postStats', postId],
    queryFn: async () => {
      const res = await client.get(`/interactions/stats/${postId}`);
      return res.data;
    },
    initialData: initialStats,
    staleTime: 1000 * 30, // 30 seconds
  });
};

export const usePostComments = (postId: string) => {
  return useQuery({
    queryKey: ['comments', postId],
    queryFn: async () => {
      const res = await client.get(`/comments/${postId}`);
      let fetchedComments = Array.isArray(res.data) ? res.data : [];
      
      const userIds = [...new Set(fetchedComments.map((c: any) => c.user_id))].filter(Boolean);
      if (userIds.length > 0) {
        const params = new URLSearchParams();
        userIds.forEach((id: any) => params.append('user_ids', String(id)));
        const usersRes = await client.get(`/users/batch?${params.toString()}`);
        const userMap: Record<string, any> = {};
        usersRes.data.forEach((u: any) => { userMap[u.user_id] = u; });
        return fetchedComments.map((c: any) => ({
          ...c,
          author: userMap[c.user_id] || { nickname: '유저' }
        }));
      }
      return fetchedComments;
    },
  });
};

export const useCollections = (enabled: boolean = false) => {
  return useQuery({
    queryKey: ['collections'],
    queryFn: async () => {
      const res = await client.get('/collections'); // Remove trailing slash to be safe
      return Array.isArray(res.data) ? res.data : [];
    },
    enabled,
    staleTime: 0,
    refetchOnWindowFocus: true,
  });
};

export const useToggleLikeMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (postId: string) => {
      const res = await client.post(`/interactions/${postId}/like`);
      return res.data;
    },
    onSuccess: (data, postId) => {
      queryClient.invalidateQueries({ queryKey: ['postStats', postId] });
      queryClient.invalidateQueries({ queryKey: ['feed'] });
      queryClient.invalidateQueries({ queryKey: ['likedIds'] });
    },
  });
};

export const useCommentMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ postId, content }: { postId: string; content: string }) => {
      return client.post('/comments/', { post_id: postId, content });
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['comments', variables.postId] });
      queryClient.invalidateQueries({ queryKey: ['feed'] });
    },
  });
};

export const useFeed = (viewMode: string, userId?: string, options: any = {}) => {
  return useInfiniteQuery({
    queryKey: ['feed', viewMode, userId, options.collectionId],
    queryFn: async ({ pageParam = 0 }) => {
      let fetchedPosts;
      const limit = 20;
      const skip = typeof pageParam === 'number' ? pageParam : 0;
      
      if (viewMode === 'all') {
        const recoRes = await client.get('/discovery/recommend', { params: { user_id: userId || '0', limit, skip } });
        const postIds = Array.isArray(recoRes.data) ? recoRes.data : [];
        if (postIds.length > 0) {
          const detailRes = await client.post('/upload/content/batch', { post_ids: postIds });
          fetchedPosts = detailRes.data;
        } else {
          fetchedPosts = [];
        }
      } else if (viewMode === 'explore') {
        fetchedPosts = (await client.get('/upload/content/feed', { params: { skip, limit } })).data;
      } else if (viewMode === 'profile' && userId && userId !== '0') {
        const res = await client.get(`/upload/content/users/${userId}/posts`, { params: { skip, limit } });
        fetchedPosts = res.data;
      } else if (viewMode === 'liked') {
        const idsRes = await client.get('/interactions/me/liked');
        const postIds = Array.isArray(idsRes.data) ? idsRes.data : [];
        if (postIds.length > 0) {
          const slicedIds = postIds.slice(skip, skip + limit);
          const detailRes = await client.post('/upload/content/batch', { post_ids: slicedIds });
          fetchedPosts = detailRes.data;
        } else {
          fetchedPosts = [];
        }
      } else if (viewMode === 'saved') {
        const idsRes = await client.get('/collections/ids');
        const postIds = Array.isArray(idsRes.data) ? idsRes.data : [];
        if (postIds.length > 0) {
          const slicedIds = postIds.slice(skip, skip + limit);
          const detailRes = await client.post('/upload/content/batch', { post_ids: slicedIds });
          fetchedPosts = detailRes.data;
        } else {
          fetchedPosts = [];
        }
      } else {
        fetchedPosts = (await client.get('/upload/content/feed', { params: { skip, limit } })).data;
      }
      
      const postsToProcess = Array.isArray(fetchedPosts) ? fetchedPosts : [];
      const userIds = [...new Set(postsToProcess.map((p: any) => p.user_id))].filter(Boolean);
      
      let processedPosts = postsToProcess;
      if (userIds.length > 0) {
        const params = new URLSearchParams();
        userIds.forEach(id => params.append('user_ids', String(id)));
        const usersRes = await client.get(`/users/batch?${params.toString()}`);
        const userMap: Record<string, any> = {};
        usersRes.data.forEach((u: any) => { userMap[u.user_id] = u; });
        processedPosts = postsToProcess.map((p: any) => ({ ...p, author: userMap[p.user_id] || { nickname: '유저' } }));
      }
      
      return {
        posts: processedPosts,
        nextSkip: processedPosts.length === limit ? skip + limit : undefined
      };
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage) => lastPage.nextSkip,
  });
};

export const useSearch = (query: string) => {
  return useQuery({
    queryKey: ['search', query],
    queryFn: async () => {
      if (!query.trim()) return [];
      
      // 1. Get IDs from recommendation service (use discovery endpoint)
      const res = await client.get('/discovery', { params: { query } });
      const postIds = Array.isArray(res.data) ? res.data : [];
      
      if (postIds.length === 0) return [];
      
      // 2. Fetch full post details from upload service (Consistent path: /upload/content/batch)
      const detailRes = await client.post('/upload/content/batch', { post_ids: postIds });
      const results = Array.isArray(detailRes.data) ? detailRes.data : [];
      
      // 3. Process authors
      const userIds = [...new Set(results.map((p: any) => p.user_id))].filter(Boolean);
      if (userIds.length > 0) {
        const params = new URLSearchParams();
        userIds.forEach(id => params.append('user_ids', String(id)));
        const usersRes = await client.get(`/users/batch?${params.toString()}`);
        const userMap: Record<string, any> = {};
        usersRes.data.forEach((u: any) => { userMap[u.user_id] = u; });
        return results.map((p: any) => ({ ...p, author: userMap[p.user_id] || { nickname: '유저' } }));
      }
      return results;
    },
    enabled: query.length > 0,
  });
};

export const useImageSearch = (imageFile: File | null, userId?: string) => {
  return useQuery({
    queryKey: ['imageSearch', imageFile?.name + '_' + imageFile?.size, userId],
    queryFn: async () => {
      if (!imageFile) return [];
      
      const formData = new FormData();
      formData.append('file', imageFile);
      
      // 1. Post to image search endpoint in recommendation service
      const res = await client.post('/discovery/image', formData, {
        params: { user_id: userId || '0' },
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      const postIds = Array.isArray(res.data) ? res.data : [];
      
      if (postIds.length === 0) return [];
      
      // 2. Fetch full post details from upload service
      const detailRes = await client.post('/upload/content/batch', { post_ids: postIds });
      const results = Array.isArray(detailRes.data) ? detailRes.data : [];
      
      // 3. Process authors
      const userIds = [...new Set(results.map((p: any) => p.user_id))].filter(Boolean);
      if (userIds.length > 0) {
        const params = new URLSearchParams();
        userIds.forEach(id => params.append('user_ids', String(id)));
        const usersRes = await client.get(`/users/batch?${params.toString()}`);
        const userMap: Record<string, any> = {};
        usersRes.data.forEach((u: any) => { userMap[u.user_id] = u; });
        return results.map((p: any) => ({ ...p, author: userMap[p.user_id] || { nickname: '유저' } }));
      }
      return results;
    },
    enabled: !!imageFile,
  });
};

export const useHashtags = (q: string) => {
  return useQuery({
    queryKey: ['hashtags', q],
    queryFn: async () => {
      const res = await client.get('/upload/content/hashtags', { params: { q, limit: 30 } });
      return Array.isArray(res.data) ? res.data as { tag: string; count: number }[] : [];
    },
    staleTime: 1000 * 60 * 5,
  });
};

export const useHashtagSearch = (tag: string) => {
  return useQuery({
    queryKey: ['hashtagSearch', tag],
    queryFn: async () => {
      if (!tag) return [];
      const res = await client.get(`/upload/content/by-hashtag/${encodeURIComponent(tag)}`);
      const results = Array.isArray(res.data) ? res.data : [];
      const userIds = [...new Set(results.map((p: any) => p.user_id))].filter(Boolean);
      if (userIds.length > 0) {
        const params = new URLSearchParams();
        userIds.forEach((id: any) => params.append('user_ids', String(id)));
        const usersRes = await client.get(`/users/batch?${params.toString()}`);
        const userMap: Record<string, any> = {};
        usersRes.data.forEach((u: any) => { userMap[u.user_id] = u; });
        return results.map((p: any) => ({ ...p, author: userMap[p.user_id] || { nickname: '유저' } }));
      }
      return results;
    },
    enabled: tag.length > 0,
  });
};

export const useSavedIds = (isLoggedIn: boolean) => {
  return useQuery({
    queryKey: ['savedIds'],
    queryFn: async () => {
      try {
        const res = await client.get('/collections/ids');
        return Array.isArray(res.data) ? res.data : [];
      } catch (err) {
        console.error("Failed to fetch saved ids:", err);
        return [];
      }
    },
    enabled: isLoggedIn,
    initialData: [],
  });
};

export const useLikedIds = (isLoggedIn: boolean) => {
  return useQuery({
    queryKey: ['likedIds'],
    queryFn: async () => {
      try {
        const res = await client.get('/interactions/me/liked');
        return Array.isArray(res.data) ? res.data : [];
      } catch (err) {
        console.error("Failed to fetch liked ids:", err);
        return [];
      }
    },
    enabled: isLoggedIn,
    initialData: [],
  });
};
