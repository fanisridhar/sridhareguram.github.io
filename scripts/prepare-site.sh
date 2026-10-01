#!/usr/bin/env bash
# Replaces the __BASE_URL__ placeholder in the pages and writes sitemap.xml and robots.txt.
# Usage: scripts/prepare-site.sh <site-dir> <base-url, no trailing slash>
set -euo pipefail

dir="${1:?site dir required}"
base="${2:?base url required}"
base="${base%/}"

for f in "$dir"/*.html; do
  sed -i "s|__BASE_URL__|${base}|g" "$f"
done

{
  echo '<?xml version="1.0" encoding="UTF-8"?>'
  echo '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
  for page in "" experience.html projects.html contact.html; do
    echo "  <url><loc>${base}/${page}</loc></url>"
  done
  echo '</urlset>'
} > "$dir/sitemap.xml"

printf 'User-agent: *\nAllow: /\nSitemap: %s/sitemap.xml\n' "$base" > "$dir/robots.txt"
echo "Prepared $dir for $base"
