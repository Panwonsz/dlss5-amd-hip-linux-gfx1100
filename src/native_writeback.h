#pragma once
#include "native_submitted_readback.h"
#include "native_game_submission.h"
#include "native_device_identity.h"
#include <vector>
#include <cstring>
/* UNVERIFIED OFFLINE DIAGNOSTIC ONLY: inverse of NativeReadSubmittedFrame.
   Upload packed pixels at the target texture's size. Submit/Flush below retain
   existing synchronous waits; this is NOT a validated Present/ECL live path.
   Caller must establish producer submission, queue/device affinity, exact bpp,
   resource states and target lifetime (including timeout). No cancellation is
   implied by an exception. See linux/build/integration-audit/REPORT.md. */
inline void NativeWriteSubmittedFrame(ID3D12CommandQueue*q,ID3D12Resource*target,D3D12_RESOURCE_STATES before,
                                      const unsigned char*pixels,size_t bpp){
 if(!q||!target||!pixels||q->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT)throw std::runtime_error("writeback queue/target");
 auto desc=target->GetDesc();
 const UINT w=UINT(desc.Width),h=desc.Height;
 if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||w<16||h<16||w>7680||h>4320||!NativeIsGameColor(desc.Format))throw std::runtime_error("writeback geometry");
 auto check=[](HRESULT hr){if(FAILED(hr))throw std::runtime_error("writeback HRESULT="+std::to_string(unsigned(hr)));};
 ID3D12Device*d=nullptr;check(q->GetDevice(IID_PPV_ARGS(&d)));
 D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT64 bytes;d->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
 D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_UPLOAD;
 D3D12_RESOURCE_DESC bd{};bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;bd.Width=bytes;bd.Height=1;bd.DepthOrArraySize=bd.MipLevels=1;bd.SampleDesc.Count=1;bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
 ID3D12Resource*upload=nullptr;HRESULT hr=NativeCreateCommittedResource(d,&hp,D3D12_HEAP_FLAG_NONE,&bd,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload));d->Release();check(hr);
 void*p=nullptr;D3D12_RANGE none{};check(upload->Map(0,&none,&p));
 for(UINT y=0;y<h;y++)std::memcpy(static_cast<unsigned char*>(p)+fp.Offset+size_t(y)*fp.Footprint.RowPitch,pixels+size_t(y)*w*bpp,size_t(w)*bpp);
 upload->Unmap(0,nullptr);
 NativeGameSubmission submit;submit.Create(q);
 submit.Submit([&](ID3D12GraphicsCommandList*c){
  D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition={target,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,D3D12_RESOURCE_STATE_COPY_DEST};
  if(before!=D3D12_RESOURCE_STATE_COPY_DEST)c->ResourceBarrier(1,&b);
  D3D12_TEXTURE_COPY_LOCATION dst{},src{};dst.pResource=target;dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.pResource=upload;src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;
  c->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
  std::swap(b.Transition.StateBefore,b.Transition.StateAfter);
  if(before!=D3D12_RESOURCE_STATE_COPY_DEST)c->ResourceBarrier(1,&b);
 });
 submit.Flush();upload->Release();
}
