FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

# --host exposes the dev server outside the container.
CMD ["npm", "run", "dev"]
