#!/usr/bin/env bash
# Install nginx virtual host + TLS for escriptorium.sigurdurhaukur.com
# Run with sudo:  sudo bash /home/haukur/escriptorium/deploy/install-nginx.sh
set -euo pipefail

DOMAIN="escriptorium.sigurdurhaukur.com"
DEPLOY="/home/haukur/escriptorium/deploy/nginx"

# Websocket upgrade map (must live in http{} context)
install -m 0644 "$DEPLOY/websocket-map.conf" "/etc/nginx/conf.d/websocket-map.conf"

install -m 0644 "$DEPLOY/${DOMAIN}.conf" "/etc/nginx/sites-available/${DOMAIN}"
ln -sf "/etc/nginx/sites-available/${DOMAIN}" "/etc/nginx/sites-enabled/${DOMAIN}"

certbot --nginx -d "$DOMAIN" \
    --non-interactive --agree-tos \
    -m haukurbirgisson5@gmail.com --no-eff-email

nginx -t
systemctl reload nginx

echo "Done: https://${DOMAIN} -> http://127.0.0.1:6005"