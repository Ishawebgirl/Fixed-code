import { Worker, Viewer } from '@react-pdf-viewer/core'
import React, { useState, useEffect, useCallback, useRef } from 'react'
import '@react-pdf-viewer/core/lib/styles/index.css'
import '@react-pdf-viewer/highlight/lib/styles/index.css'
import '@react-pdf-viewer/default-layout/lib/styles/index.css'
import '@react-pdf-viewer/page-navigation/lib/styles/index.css'
import { RenderHighlightsProps, Trigger } from '@react-pdf-viewer/highlight'
import { HighlightArea } from '@react-pdf-viewer/highlight'
import { highlightPlugin } from '@react-pdf-viewer/highlight'
import { scrollModePlugin } from '@react-pdf-viewer/scroll-mode'
import { ScrollMode } from '@react-pdf-viewer/core'
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout'
import { pageNavigationPlugin } from '@react-pdf-viewer/page-navigation'
import { Citation } from 'api/model'
import * as Model from 'api/model'
import Icon from './Icon'

const multiplierX = 11.75
const multiplierY = 9.08
const paddingLeft = 0.4
const paddingRight = 0.5
const paddingTop = 0.2
const paddingBottom = 0.2

interface IDocumentViewer {
  theDocument: Model.Document | undefined
  claimUniqueID: string | undefined
  citations: Model.Citation[] | undefined
  onCancel: any
}

const DocumentViewer = ({
  theDocument,
  claimUniqueID,
  citations,
  onCancel,
}: IDocumentViewer) => {
  const [areas, setAreas] = useState<HighlightArea[]>([])
  const [initialPageIndex, setInitialPageIndex] = useState<number>(0)

  const renderHighlights = (props: RenderHighlightsProps) => (
    <div>
      {areas?.length >= 0 &&
        areas!
          .filter((area: { pageIndex: number }) => area.pageIndex === props.pageIndex)
          .map((area: HighlightArea, idx: React.Key | null | undefined) => (
            <div
              key={idx}
              className="highlight-area"
              style={Object.assign({}, props.getCssProperties(area, props.rotation))}
            />
          ))}
    </div>
  )

  const highlightPluginInstance = highlightPlugin({
    trigger: Trigger.None,
    renderHighlights,
  })

  const scrollModePluginInstance = scrollModePlugin()
  scrollModePluginInstance.switchScrollMode(ScrollMode.Wrapped)

  const applyHighlight = (citation: Citation) => {
    const citationAreas: HighlightArea[] = []
    const boundingBoxes = citation.boundingBoxes.split('|')

    for (const boundingBox of boundingBoxes) {
      const coords = boundingBox.split(',')
      const top = parseFloat(coords[1]) * multiplierY - paddingTop
      const left = parseFloat(coords[0]) * multiplierX - paddingLeft
      const height =
        parseFloat(coords[3]) * multiplierY - top + paddingTop + paddingBottom
      const width =
        parseFloat(coords[2]) * multiplierX - left + paddingLeft + paddingRight

      const area = {
        pageIndex: citation.pageNumber - 1,
        top,
        left,
        height,
        width,
      } as HighlightArea

      citationAreas.push(area)
    }

    return citationAreas
  }

  const applyHighlights = (citations: Model.Citation[]) => {
    const allAreas: HighlightArea[] = []

    for (const citation of citations) {
      if (citation.documentUniqueID === theDocument!.uniqueID) {
        const citationAreas = applyHighlight(citation)
        allAreas.push(...citationAreas)
      }
    }

    setAreas(allAreas)
  }

  useEffect(() => {
    if (!citations || citations.length === 0 || theDocument === undefined) {
      return
    }

    applyHighlights(citations)
    setInitialPageIndex(citations[0].pageNumber - 1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [citations, theDocument])

  const escFunction = useCallback(
    (event: { key: string }) => {
      if (event.key === 'Escape') {
        onCancel()
      }
    },
    [onCancel]
  )

  const useOutsideAlerter = (ref: React.RefObject<HTMLElement>) => {
    useEffect(() => {
      if (theDocument === undefined) {
        return
      }
      const handleClickOutside = (event: { target: any }) => {
        if (ref.current && !ref.current.contains(event.target)) {
          onCancel()
        }
      }
      document.addEventListener('mousedown', handleClickOutside)
      return () => {
        document.removeEventListener('mousedown', handleClickOutside)
      }
    }, [ref])
  }

  useEffect(() => {
    document.addEventListener('keydown', escFunction, false)

    return () => {
      document.removeEventListener('keydown', escFunction, false)
    }
  }, [escFunction])

  const defaultLayoutPluginInstance = defaultLayoutPlugin()
  const pageNavigationPluginInstance = pageNavigationPlugin()
  const { jumpToPage } = pageNavigationPluginInstance

  const navRef = useRef(null)
  useOutsideAlerter(navRef)

  return (
    <nav
      ref={navRef}
      id="document-viewer"
      className={`document-viewer${theDocument !== undefined ? '' : ' collapsed'}`}>
      <div className="collapse-button" onClick={onCancel}>
        <Icon name="AngleDoubleRight"></Icon>
      </div>
      <div className="caption">{theDocument?.name}</div>
      <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
        <div style={{ height: 'calc(100vh - 120px)' }}>
          <Viewer
            plugins={[
              defaultLayoutPluginInstance,
              pageNavigationPluginInstance,
              highlightPluginInstance,
              scrollModePluginInstance,
            ]}
            withCredentials={true}
            initialPage={initialPageIndex}
            onDocumentLoad={() => {
              setTimeout(() => {
                jumpToPage(initialPageIndex)
              }, 1000)
            }}
            fileUrl={`${process.env.REACT_APP_API_URL}/claims/${claimUniqueID}/documents/${theDocument?.uniqueID}`}
          />
        </div>
      </Worker>
    </nav>
  )
}

export default DocumentViewer
