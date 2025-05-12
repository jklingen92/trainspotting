from django.urls import path

from detection.views import CameraPageView, CameraStreamView, SnapshotView

urlpatterns = [
    path('camera/', CameraPageView.as_view(), name='camera_page'),
    path('camera/stream/', CameraStreamView.as_view(), name='camera_stream'),
    path('camera/snap/', SnapshotView.as_view(), name='snapshot_view'),
]