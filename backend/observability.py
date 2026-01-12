import os
import sys
import json
from typing import List, Dict, Any
import google.generativeai as genai
from langchain.docstore.document import Document
from collections import deque, defaultdict
import time


# Add config directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class ObservabilityMetrics:
    """Track and analyze RAG system performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = {
            'requests': deque(maxlen=max_history),
            'latencies': deque(maxlen=max_history),
            'confidence_scores': deque(maxlen=max_history),
            'document_counts': deque(maxlen=max_history),
            'search_types': defaultdict(int),
            'error_types': defaultdict(int),
            'hourly_stats': defaultdict(lambda: {'count': 0, 'avg_latency': 0, 'errors': 0})
        }
        self.start_time = time.time()
    
    def record_request(self, request_data: Dict[str, Any]):
        """Record a request and its metrics"""
        timestamp = time.time()
        hour_key = int(timestamp // 3600)
        
        # Extract metrics
        latency = request_data.get('latency', 0)
        confidence = request_data.get('confidence_score', 0)
        doc_count = request_data.get('document_count', 0)
        search_type = request_data.get('search_type', 'unknown')
        error_type = request_data.get('error_type')
        not_found = request_data.get('not_found', False)
        
        # Record basic metrics
        self.metrics['requests'].append({
            'timestamp': timestamp,
            'latency': latency,
            'confidence': confidence,
            'doc_count': doc_count,
            'search_type': search_type,
            'not_found': not_found,
            'error': error_type is not None
        })
        
        self.metrics['latencies'].append(latency)
        self.metrics['confidence_scores'].append(confidence)
        self.metrics['document_counts'].append(doc_count)
        self.metrics['search_types'][search_type] += 1
        
        if error_type:
            self.metrics['error_types'][error_type] += 1
        
        # Update hourly stats
        hourly = self.metrics['hourly_stats'][hour_key]
        hourly['count'] += 1
        if hourly['count'] == 1:
            hourly['avg_latency'] = latency
        else:
            hourly['avg_latency'] = (hourly['avg_latency'] * (hourly['count'] - 1) + latency) / hourly['count']
        
        if error_type:
            hourly['errors'] += 1
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.metrics['latencies']:
            return self._empty_stats()
        
        # Calculate percentiles
        latencies = list(self.metrics['latencies'])
        confidences = list(self.metrics['confidence_scores'])
        
        return {
            'uptime_seconds': time.time() - self.start_time,
            'total_requests': len(self.metrics['requests']),
            'requests_per_minute': self._calculate_rpm(),
            'latency': {
                'avg': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': self._percentile(latencies, 95),
                'p99': self._percentile(latencies, 99),
                'min': min(latencies),
                'max': max(latencies)
            },
            'confidence': {
                'avg': statistics.mean(confidences),
                'median': statistics.median(confidences),
                'min': min(confidences),
                'max': max(confidences)
            },
            'documents_per_request': {
                'avg': statistics.mean(list(self.metrics['document_counts'])),
                'median': statistics.median(list(self.metrics['document_counts']))
            },
            'search_types': dict(self.metrics['search_types']),
            'error_rate': self._calculate_error_rate(),
            'not_found_rate': self._calculate_not_found_rate(),
            'recent_performance': self._get_recent_performance()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        stats = self.get_current_stats()
        
        # Health checks
        health_checks = {
            'latency_healthy': stats['latency']['p95'] < 5000,  # < 5 seconds
            'confidence_healthy': stats['confidence']['avg'] > 0.5,  # > 50% confidence
            'error_rate_healthy': stats['error_rate'] < 0.05,  # < 5% error rate
            'not_found_healthy': stats['not_found_rate'] < 0.2,  # < 20% not found
            'response_rate_healthy': stats['requests_per_minute'] > 0  # Getting requests
        }
        
        overall_health = all(health_checks.values())
        
        return {
            'healthy': overall_health,
            'status': 'healthy' if overall_health else 'degraded',
            'checks': health_checks,
            'summary': stats
        }
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON string or file"""
        data = {
            'timestamp': time.time(),
            'stats': self.get_current_stats(),
            'health': self.get_health_status(),
            'raw_metrics': {
                'requests': list(self.metrics['requests'])[-100:],  # Last 100 requests
                'search_types': dict(self.metrics['search_types']),
                'error_types': dict(self.metrics['error_types'])
            }
        }
        
        json_str = json.dumps(data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty stats structure"""
        return {
            'uptime_seconds': 0,
            'total_requests': 0,
            'requests_per_minute': 0,
            'latency': {'avg': 0, 'median': 0, 'p95': 0, 'p99': 0, 'min': 0, 'max': 0},
            'confidence': {'avg': 0, 'median': 0, 'min': 0, 'max': 0},
            'documents_per_request': {'avg': 0, 'median': 0},
            'search_types': {},
            'error_rate': 0,
            'not_found_rate': 0,
            'recent_performance': []
        }
    
    def _calculate_rpm(self) -> float:
        """Calculate requests per minute"""
        if not self.metrics['requests']:
            return 0
        
        now = time.time()
        recent_requests = [r for r in self.metrics['requests'] if now - r['timestamp'] < 60]
        return len(recent_requests)
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        if not self.metrics['requests']:
            return 0
        
        errors = sum(1 for r in self.metrics['requests'] if r.get('error', False))
        return errors / len(self.metrics['requests'])
    
    def _calculate_not_found_rate(self) -> float:
        """Calculate not found rate"""
        if not self.metrics['requests']:
            return 0
        
        not_found = sum(1 for r in self.metrics['requests'] if r.get('not_found', False))
        return not_found / len(self.metrics['requests'])
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _get_recent_performance(self) -> List[Dict[str, Any]]:
        """Get recent performance data"""
        recent = list(self.metrics['requests'])[-10:]  # Last 10 requests
        return [
            {
                'timestamp': r['timestamp'],
                'latency': r['latency'],
                'confidence': r['confidence'],
                'doc_count': r['doc_count'],
                'search_type': r['search_type']
            }
            for r in recent
        ]

# Global observability instance
observability = ObservabilityMetrics()
