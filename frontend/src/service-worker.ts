const CACHE_NAME = 'jarvis-cache-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/index.html',
  '/manifest.json',
  '/offline.html',
  '/favicon.ico'
];

// @ts-ignore
self.addEventListener('install', (event) => {
  // @ts-ignore
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS_TO_CACHE))
      // @ts-ignore
      .then(() => self.skipWaiting())
  );
});

// @ts-ignore
self.addEventListener('activate', (event) => {
  // @ts-ignore
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
      // @ts-ignore
    }).then(() => self.clients.claim())
  );
});

// @ts-ignore
self.addEventListener('fetch', (event) => {
  // @ts-ignore
  if (event.request.method !== 'GET') return;

  // @ts-ignore
  event.respondWith(
    // @ts-ignore
    caches.match(event.request).then((response) => {
      if (response) return response;

      // @ts-ignore
      return fetch(event.request).then((response) => {
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        const responseToCache = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          // @ts-ignore
          cache.put(event.request, responseToCache);
        });

        return response;
      }).catch(() => {
        // @ts-ignore
        if (event.request.mode === 'navigate') {
          return caches.match('/offline.html');
        }
        return new Response('Offline', { status: 503 });
      });
    })
  );
});
