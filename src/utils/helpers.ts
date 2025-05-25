import { twMerge } from 'tailwind-merge';
import { clsx, type ClassValue } from 'clsx';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function truncateFilename(filename: string, maxLength = 20): string {
  if (filename.length <= maxLength) return filename;
  
  const extension = filename.split('.').pop() || '';
  const name = filename.substring(0, filename.length - extension.length - 1);
  
  if (name.length <= maxLength - 3 - extension.length) {
    return filename;
  }
  
  return `${name.substring(0, maxLength - 3 - extension.length)}...${extension ? `.${extension}` : ''}`;
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export function getPercentage(value: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((value / total) * 100);
}

export function isDatasetFile(fileType: string): boolean {
  return ['csv', 'xlsx'].includes(fileType);
}

export function isDocumentFile(fileType: string): boolean {
  return ['pdf', 'text'].includes(fileType);
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function getRandomColor(index: number): string {
  const colors = [
    'rgb(255, 99, 132)', // red
    'rgb(54, 162, 235)', // blue
    'rgb(255, 206, 86)', // yellow
    'rgb(75, 192, 192)', // green
    'rgb(153, 102, 255)', // purple
    'rgb(255, 159, 64)', // orange
    'rgb(199, 199, 199)', // gray
    'rgb(83, 102, 255)', // indigo
    'rgb(255, 99, 255)', // pink
    'rgb(0, 201, 151)', // teal
  ];
  
  return colors[index % colors.length];
}