# Bee-A-Hero frontend (React/Vite) — built to static assets, served by nginx.
# Build context = repo ROOT.  VITE_API_URL is baked at build time (the browser calls it
# directly), so point it at the backend's PUBLIC url.
#   docker build -f docker/frontend.Dockerfile --build-arg VITE_API_URL=http://localhost:8000 -t bee-frontend .
FROM node:20-alpine AS build
WORKDIR /app
COPY bee-a-hero-app/frontend/package*.json ./
RUN npm install
COPY bee-a-hero-app/frontend/ ./
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
