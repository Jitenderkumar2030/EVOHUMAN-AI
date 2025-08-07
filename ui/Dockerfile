# Multi-stage build for React UI
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user
RUN addgroup -g 1001 -S evohuman && \
    adduser -S evohuman -u 1001

# Set ownership
RUN chown -R evohuman:evohuman /usr/share/nginx/html
RUN chown -R evohuman:evohuman /var/cache/nginx
RUN chown -R evohuman:evohuman /var/log/nginx
RUN chown -R evohuman:evohuman /etc/nginx/conf.d
RUN touch /var/run/nginx.pid
RUN chown -R evohuman:evohuman /var/run/nginx.pid

USER evohuman

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
