nswag run nswag.json
echo "import { ApiBase } from './base'" | cat - model.ts > temp && mv temp model.ts
