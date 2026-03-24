import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'
import type { TreeNode } from '../types'

interface FlatNode {
  id: string
  label: string
  visits: number
  q: Record<string, number>
  description: string
  prior: number
  depth: number
}

interface FlatLink {
  source: string
  target: string
}

function flatten(tree: TreeNode, parentId: string = '', depth: number = 0): { nodes: FlatNode[]; links: FlatLink[] } {
  const id = parentId ? `${parentId}>${tree.action}` : 'root'
  const nodes: FlatNode[] = [{
    id,
    label: tree.action,
    visits: tree.visits,
    q: tree.q || {},
    description: tree.description || '',
    prior: tree.prior || 0,
    depth,
  }]
  const links: FlatLink[] = []

  for (const child of tree.children || []) {
    links.push({ source: id, target: `${id}>${child.action}` })
    const sub = flatten(child, id, depth + 1)
    nodes.push(...sub.nodes)
    links.push(...sub.links)
  }
  return { nodes, links }
}

export default function TreeView({ tree }: { tree: TreeNode | null }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selected, setSelected] = useState<FlatNode | null>(null)

  const render = useCallback(() => {
    if (!svgRef.current || !tree) return

    const { nodes, links } = flatten(tree)
    const svg = d3.select(svgRef.current)
    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight

    svg.selectAll('g.content').remove()
    const g = svg.append('g').attr('class', 'content')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (e) => g.attr('transform', e.transform))
    svg.call(zoom)

    const qColor = (node: FlatNode) => {
      const values = Object.values(node.q)
      if (values.length === 0) return '#6b7280'
      const avg = values.reduce((a, b) => a + b, 0) / values.length
      return d3.interpolateRdYlGn(avg)
    }

    const sim = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links as any).id((d: any) => d.id).distance(60).strength(0.8))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('y', d3.forceY(0).strength((d: any) => d.depth * 0.02))

    const link = g.selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#374151')
      .attr('stroke-width', 1)

    const node = g.selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d: any) => Math.max(4, Math.log(d.visits + 1) * 4))
      .attr('fill', (d: any) => qColor(d))
      .attr('stroke', (d: any) => d.id === 'root' ? '#f59e0b' : '#1f2937')
      .attr('stroke-width', (d: any) => d.id === 'root' ? 3 : 1)
      .attr('cursor', 'pointer')
      .on('click', (_: any, d: any) => setSelected(d))
      .call(d3.drag<any, any>()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null })
      )

    const labels = g.selectAll('text')
      .data(nodes)
      .join('text')
      .text((d: any) => d.label)
      .attr('font-size', 9)
      .attr('fill', '#9ca3af')
      .attr('text-anchor', 'middle')
      .attr('dy', (d: any) => -Math.max(4, Math.log(d.visits + 1) * 4) - 4)

    sim.on('tick', () => {
      link.attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y)
      node.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y)
      labels.attr('x', (d: any) => d.x).attr('y', (d: any) => d.y)
    })

    setTimeout(() => {
      const root = nodes[0] as any
      if (root?.x != null) {
        svg.transition().duration(500).call(
          zoom.transform,
          d3.zoomIdentity.translate(width / 2 - root.x, height / 2 - root.y)
        )
      }
    }, 300)
  }, [tree])

  useEffect(() => { render() }, [render])

  return (
    <div className="w-96 border-l border-gray-800 flex flex-col bg-gray-950">
      <div className="p-3 border-b border-gray-800 text-sm font-medium text-gray-400">
        Search Tree
      </div>

      <div className="flex-1 relative">
        {tree ? (
          <svg ref={svgRef} className="w-full h-full" />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm text-center px-4">
            <div>
              <div className="text-2xl mb-2">🌳</div>
              Run an analysis to see<br />the search tree
            </div>
          </div>
        )}
      </div>

      {/* Node detail panel — full expandable description */}
      {selected && <NodeDetail node={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}

function NodeDetail({ node, onClose }: { node: FlatNode; onClose: () => void }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border-t border-gray-800 text-xs max-h-72 overflow-y-auto">
      {/* Header */}
      <div className="p-3 pb-2 flex items-start justify-between gap-2">
        <div className="font-medium text-blue-300 text-sm">{node.label}</div>
        <button onClick={onClose} className="text-gray-600 hover:text-gray-400 flex-shrink-0">✕</button>
      </div>

      <div className="px-3 pb-3 space-y-2">
        {/* Description — expandable */}
        {node.description && (
          <div>
            <div
              className={`text-gray-300 leading-relaxed ${expanded ? '' : 'line-clamp-3'}`}
              style={!expanded ? { display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' } : {}}
            >
              {node.description}
            </div>
            {node.description.length > 120 && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-blue-400 hover:text-blue-300 mt-1"
              >
                {expanded ? '▲ Collapse' : '▼ Expand full description'}
              </button>
            )}
          </div>
        )}

        {/* Stats */}
        <div className="flex gap-3 text-gray-500">
          <span>Visits: {node.visits}</span>
          <span>Prior: {(node.prior * 100).toFixed(0)}%</span>
          <span>Depth: {node.depth}</span>
        </div>

        {/* Q-values per player */}
        {Object.keys(node.q).length > 0 && (
          <div className="space-y-1 pt-1 border-t border-gray-800">
            <div className="text-gray-500 font-medium">Player Scores</div>
            {Object.entries(node.q).map(([player, val]) => (
              <div key={player} className="flex items-center gap-2">
                <span className="text-gray-400 w-28 truncate">{player}</span>
                <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${val * 100}%`, background: d3.interpolateRdYlGn(val) }}
                  />
                </div>
                <span className="w-10 text-right" style={{ color: d3.interpolateRdYlGn(val) }}>
                  {(val * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
